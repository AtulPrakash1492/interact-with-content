export default async function handler(req, res) {
    if(req.method === 'POST') {
        const response= await fetch('http://localhost:5000/content', {
            method: 'POST',
            headers: { 'Content-Type': 'spplication/json' },
            body: JSON.stringify(req.body)
        });

        const data = await response.json();
        res.status(200).json(data);
    }
    else {
        res.status(405).json({ messsage: 'Method not allowed' });
    }
}