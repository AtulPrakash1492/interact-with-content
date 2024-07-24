import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMProcessor:
    """
    A processor class for the Mistral-v0.2 INSTRUCT large language model.
    It initializes the model and tokenizer and provides
    methods to process prompts using Mistral-v0.2 INSTRUCT.

    Attributes:
        device (str): The device to run the model on ('cuda' or 'cpu').
        tokenizer (AutoTokenizer): The tokenizer for Mistral-v0.2 INSTRUCT.
        model (AutoModelForCausalLM): The language model.
        chat_history (list): A list of chat messages to maintain the conversation context.
        max_new_tokens (int): Maximum number of new tokens to generate (fixed at 1000).
        debug_mode (bool): Indicates whether debug information should be printed.
    """

    def __init__(self, device='cuda', debug_mode=False):
        
        self.max_new_tokens = 1000  # Fixed maximum new tokens
        self.device = device
        self.debug_mode = debug_mode
        self.chat_history = []
        self.initialize_model()

    def initialize_model(self):
        
        """
        Initializes the tokenizer and model for Mistral-v0.2.
        """
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(self.device)
            
        except OSError as e:
            raise RuntimeError(f"Failed to download or load the model/tokenizer from {model_name}. Check your internet connection or model name.") from e
        
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load model onto device {self.device}. Ensure that the specified device is available.") from e

    def chat_with_model(self, new_prompt):
        """
        Manages the conversation by adding new prompts, encoding the chat history,
        handling token limits, and generating responses.

        Args:
            new_prompt (str): The new user input to be added to the conversation.

        Returns:
            str: The processed output from the LLM, responding to the entire conversation.
        """
        
        self.chat_history.append({"role": "user", "content": new_prompt})
        encodeds = self.tokenizer.apply_chat_template(self.chat_history, return_tensors="pt").to(self.device)

        # If the total tokens exceed context-length, remove the second pair of user-assistant messages
        while encodeds.size(1) + self.max_new_tokens > 32000:
            if len(self.chat_history) > 4:
                # Pop the second user and second assistant message
                self.chat_history.pop(1)
                self.chat_history.pop(1)
            encodeds = self.tokenizer.apply_chat_template(self.chat_history, return_tensors="pt").to(self.device)

        # Generate Response
        try:
            generated_ids = self.model.generate(
                encodeds,
                do_sample=True,
                top_k=1,
                top_p=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens
            )
            
            # Decode and update the chat history
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split('[/INST]')[-1]
            self.chat_history.append({"role": "assistant", "content": decoded})
            
            return decoded
        
        except Exception as e:
            raise Exception(f"Error during response generation: {e}")