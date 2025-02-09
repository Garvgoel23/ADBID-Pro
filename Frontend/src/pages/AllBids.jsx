import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { formatDistance } from 'date-fns';
import Card from '../components/Card';

function AllBids() {
  const [bids, setBids] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Fetch all bids from Supabase
    // Mock data for now
    const mockBids = [
      {
        id: 1,
        title: 'Vintage Watch',
        currentBid: 150,
        endTime: new Date(Date.now() + 86400000),
        image: 'https://placehold.co/200x200?text=Watch',
        bids: 5
      },
      {
        id: 2,
        title: 'Gaming Console',
        currentBid: 300,
        endTime: new Date(Date.now() + 172800000),
        image: 'https://placehold.co/200x200?text=Console',
        bids: 8
      },
      {
        id: 3,
        title: 'Antique Furniture',
        currentBid: 500,
        endTime: new Date(Date.now() + 259200000),
        image: 'https://placehold.co/200x200?text=Furniture',
        bids: 3
      }
    ];
    setBids(mockBids);
    setLoading(false);
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">All Active Bids</h1>
        <Link to="/create-bid" className="btn btn-primary">Create New Bid</Link>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* {bids.map((bid) => (
          <Link key={bid.id} to={`/bids/${bid.id}`} className="card hover:shadow-lg transition-shadow">
            <img src={bid.image} alt={bid.title} className="w-full h-48 object-cover rounded-md mb-4" />
            <h3 className="text-xl font-semibold mb-2">{bid.title}</h3>
            <div className="flex justify-between items-center text-gray-600 mb-2">
              <span>Current Bid: ${bid.currentBid}</span>
              <span>{bid.bids} bids</span>
            </div>
            <div className="text-sm text-gray-500">
              Ends {formatDistance(bid.endTime, new Date(), { addSuffix: true })}
            </div>
          </Link>
        ))} */}
        <Card/>
        

        


      </div>
    </div>
  );
}

export default AllBids