import React from 'react';
import { BarChart, DollarSign, Percent, Activity } from 'lucide-react';

export default function Dashboard() {
  const stats = [
    {
      title: 'Total Bid Requests',
      value: '2,543',
      change: '+12.5%',
      icon: <Activity className="w-6 h-6" />,
    },
    {
      title: 'Win Rate',
      value: '45.2%',
      change: '+3.2%',
      icon: <Percent className="w-6 h-6" />,
    },
    {
      title: 'Total Spend',
      value: '$12,435',
      change: '+15.3%',
      icon: <DollarSign className="w-6 h-6" />,
    },
    {
      title: 'Average CPM',
      value: '$4.23',
      change: '-2.1%',
      icon: <BarChart className="w-6 h-6" />,
    },
  ];

  return (
    <div>
      {/* <h1 className="text-2xl font-bold mb-8">Dashboard</h1> */}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => (
          <div key={stat.title} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="text-gray-500">{stat.title}</div>
              <div className="bg-blue-100 p-2 rounded-lg">{stat.icon}</div>
            </div>
            <div className="text-2xl font-semibold mb-2">{stat.value}</div>
            <div className={`text-sm ${stat.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
              {stat.change} from last month
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Bids</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b">
                  <th className="pb-3">Time</th>
                  <th className="pb-3">Publisher</th>
                  <th className="pb-3">Bid Amount</th>
                  <th className="pb-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {/* Sample data - replace with real data */}
                <tr className="border-b">
                  <td className="py-3">2 min ago</td>
                  <td>Publisher A</td>
                  <td>$2.50</td>
                  <td><span className="text-green-500">Won</span></td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">5 min ago</td>
                  <td>Publisher B</td>
                  <td>$1.75</td>
                  <td><span className="text-red-500">Lost</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Spend by Publisher</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span>Publisher A</span>
                <span>$4,235</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '70%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span>Publisher B</span>
                <span>$3,128</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '55%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}