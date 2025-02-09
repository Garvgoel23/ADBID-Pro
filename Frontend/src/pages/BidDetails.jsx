import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { formatDistance, format } from 'date-fns';
import toast from 'react-hot-toast';

function BidDetails() {
  const { id } = useParams();
  const [bid, setBid] = useState(null);
  const [bidAmount, setBidAmount] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Fetch bid details from Supabase
    // Mock data for now
    const mockBid = {
      id: parseInt(id),
      title: 'Vintage Watch',
      description: 'A beautiful vintage watch in excellent condition.',
      currentBid: 150,
      startingPrice: 100,
      endTime: new Date(Date.now() + 86400000),
      image: 'https://placehold.co/400x300?text=Watch',
      bidHistory: [
        { id: 1, amount: 150, user: 'user123', timestamp: new Date(Date.now() - 3600000) },
        { id: 2, amount: 125, user: 'user456', timestamp: new Date(Date.now() - 7200000) },
        { id: 3, amount: 100, user: 'user789', timestamp: new Date(Date.now() - 10800000) }
      ]
    };
    setBid(mockBid);
    setLoading(false);
  }, [id]);

  const handleBid = async (e) => {
    e.preventDefault();
    if (parseFloat(bidAmount) <= bid.currentBid) {
      toast.error('Bid amount must be higher than current bid');
      return;
    }
    // TODO: Implement bid submission to Supabase
    toast.success('Bid placed successfully!');
    setBidAmount('');
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  if (!bid) {
    return <div className="text-center py-12">Bid not found</div>;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <img src={bid.image} alt={bid.title} className="w-full rounded-lg shadow-md" />
        </div>
        
        <div>
          <h1 className="text-3xl font-bold mb-4">{bid.title}</h1>
          <p className="text-gray-600 mb-6">{bid.description}</p>
          
          <div className="card mb-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <p className="text-sm text-gray-500">Current Bid</p>
                <p className="text-2xl font-bold">${bid.currentBid}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Time Left</p>
                <p className="text-lg">{formatDistance(bid.endTime, new Date(), { addSuffix: true })}</p>
              </div>
            </div>
            
            <form onSubmit={handleBid} className="flex space-x-4">
              <input
                type="number"
                value={bidAmount}
                onChange={(e) => setBidAmount(e.target.value)}
                className="input flex-1"
                min={bid.currentBid + 1}
                step="0.01"
                placeholder={`Min bid: $${bid.currentBid + 1}`}
                required
              />
              <button type="submit" className="btn btn-primary">
                Place Bid
              </button>
            </form>
          </div>
          
          <div>
            <h2 className="text-xl font-semibold mb-4">Bid History</h2>
            <div className="space-y-4">
              {bid.bidHistory.map((history) => (
                <div key={history.id} className="flex justify-between items-center p-4 bg-gray-50 rounded-md">
                  <div>
                    <p className="font-medium">${history.amount}</p>
                    <p className="text-sm text-gray-500">by {history.user}</p>
                  </div>
                  <p className="text-sm text-gray-500">
                    {format(history.timestamp, 'MMM d, yyyy h:mm a')}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BidDetails