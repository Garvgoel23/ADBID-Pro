import React from "react";
import { LayoutDashboard, Gavel, LogOut, User } from 'lucide-react';
function Layout({ children }) {
	return (
		<>
			<div className="flex h-screen bg-gray-100">
				{/* Sidebar */}
				<div className="w-64 bg-white shadow-lg">
					<div className="p-4">
						<h1 className="text-2xl font-bold text-gray-800">
							AdBid Pro
						</h1>
					</div>
					<nav className="mt-8">
						<a
							// onClick={() => navigate("/dashboard")}
							className="flex items-center px-6 py-3 text-gray-700 hover:bg-gray-100 cursor-pointer"
						>
							<LayoutDashboard className="w-5 h-5 mr-3" />
							Dashboard
						</a>
						<a
							// onClick={() => navigate("/auction")}
							className="flex items-center px-6 py-3 text-gray-700 hover:bg-gray-100 cursor-pointer"
						>
							<Gavel className="w-5 h-5 mr-3" />
							Auction
						</a>
						<a
							// onClick={() => navigate("/profile")}
							className="flex items-center px-6 py-3 text-gray-700 hover:bg-gray-100 cursor-pointer"
						>
							<User className="w-5 h-5 mr-3" />
							Profile
						</a>
						<a
							// onClick={() => navigate("/logout")}
							className="flex items-center px-6 py-3 text-gray-700 hover:bg-gray-100 cursor-pointer"
						>
							{/* <LogOut className="w-5 h-5 mr-3" /> */}
							Logout
						</a>
					</nav>
				</div>

				{/* Main Content */}
				<div className="flex-1 overflow-auto">
					<div className="p-8">{children}</div>
				</div>
			</div>
		</>
	);
}

export default Layout;
