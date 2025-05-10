import React from "react";

const DashboardSimulation: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h1 className="text-3xl font-semibold text-gray-900 dark:text-white mb-4">
          Simulation Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Manage and monitor your laboratory simulations
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Active Simulations
          </h2>
          <div className="space-y-2">
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
              <p className="text-gray-900 dark:text-gray-100">
                No active simulations
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Simulation Controls
          </h2>
          <button className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors">
            Start New Simulation
          </button>
        </div>
      </div>
    </div>
  );
};

export default DashboardSimulation;
