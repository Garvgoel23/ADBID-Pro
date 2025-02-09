import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/NavBar';
import Dashboard from './pages/Dashboard';
import CreateBid from './pages/CreateBid';
import AllBids from './pages/AllBids';
import BidDetails from './pages/BidDetails';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/create-bid" element={<CreateBid />} />
            <Route path="/bids" element={<AllBids />} />
            <Route path="/bids/:id" element={<BidDetails />} />
          </Routes>
        </main>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;