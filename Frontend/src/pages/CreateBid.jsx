import { useState } from "react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";

function CreateBid() {
	const navigate = useNavigate();
	const [formData, setFormData] = useState({
		title: "",
		description: "",
		startingPrice: "",
		endDate: "",
		image: "",
	});

	const handleSubmit = async (e) => {
		e.preventDefault();
		// TODO: Implement Supabase integration
		toast.success("Bid created successfully!");
		navigate("/bids");
	};

	const handleChange = (e) => {
		const { name, value } = e.target;
		setFormData((prev) => ({ ...prev, [name]: value }));
	};

	return (
		<div className="max-w-2xl mx-auto">
			<h1 className="text-3xl font-bold mb-8">Create New Bid</h1>

			<form onSubmit={handleSubmit} className="card space-y-6">
				<div>
					<label
						htmlFor="title"
						className="block text-sm font-medium text-gray-700 mb-1"
					>
						Title
					</label>
					<input
						type="text"
						id="title"
						name="title"
						value={formData.title}
						onChange={handleChange}
						className="input w-full "
						required
					/>
				</div>

				<div>
					<label
						htmlFor="description"
						className="block text-sm font-medium text-gray-700 mb-1"
					>
						Description
					</label>
					<textarea
						id="description"
						name="description"
						value={formData.description}
						onChange={handleChange}
						rows="4"
						className="input w-full"
						required
					/>
				</div>

				<div>
					<label
						htmlFor="startingPrice"
						className="block text-sm font-medium text-gray-700 mb-1"
					>
						Starting Price ($)
					</label>
					<input
						type="number"
						id="startingPrice"
						name="startingPrice"
						value={formData.startingPrice}
						onChange={handleChange}
						min="0"
						step="0.01"
						className="input w-full"
						required
					/>
				</div>

				<div>
					<label
						htmlFor="endDate"
						className="block text-sm font-medium text-gray-700 mb-1"
					>
						End Date
					</label>
					<input
						type="datetime-local"
						id="endDate"
						name="endDate"
						value={formData.endDate}
						onChange={handleChange}
						className="input w-full"
						required
					/>
				</div>

				<div>
					<label
						htmlFor="image"
						className="block text-sm font-medium text-gray-700 mb-1"
					>
						Image URL
					</label>
					<input
						type="url"
						id="image"
						name="image"
						value={formData.image}
						onChange={handleChange}
						className="input w-full"
						placeholder="https://example.com/image.jpg"
						required
					/>
				</div>

				<div className="flex justify-end space-x-4">
					<button
						type="button"
						onClick={() => navigate(-1)}
						className="btn btn-secondary"
					>
						Cancel
					</button>
					<button type="submit" className="btn btn-primary">
						Create Bid
					</button>
				</div>
			</form>
		</div>
	);
}

export default CreateBid;
