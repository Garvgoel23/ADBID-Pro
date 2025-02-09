import { Link } from "react-router-dom";
// import { HomeIcon, PlusCircleIcon, ClipboardDocumentListIcon } from '@heroicons/react/24/outline';

function Navbar({ children }) {
	return (
		<main>
			<nav className="bg-white shadow-md">
				<div className="container mx-auto px-4">
					<div className="flex items-center justify-between h-16">
						<Link to="/" className="flex items-center space-x-2">
							<img
								src="/vite.svg"
								alt="Logo"
								className="h-8 w-8"
							/>
							<span className="text-xl font-bold text-gray-800">
								BidHub
							</span>
						</Link>

						<div className="flex space-x-4">
							<Link
								to="/"
								className="flex items-center space-x-1 text-gray-600 hover:text-blue-600"
							>
								{/* <HomeIcon className="h-5 w-5" /> */}
								<span>Dashboard</span>
							</Link>
							<Link
								to="/create-bid"
								className="flex items-center space-x-1 text-gray-600 hover:text-blue-600"
							>
								{/* <PlusCircleIcon className="h-5 w-5" /> */}
								<span>Create Bid</span>
							</Link>
							<Link
								to="/bids"
								className="flex items-center space-x-1 text-gray-600 hover:text-blue-600"
							>
								{/* <ClipboardDocumentListIcon className="h-5 w-5" /> */}
								<span>All Bids</span>
							</Link>
						</div>
					</div>
				</div>
			</nav>
			<div>{children}</div>
		</main>
	);
}

export default Navbar;
