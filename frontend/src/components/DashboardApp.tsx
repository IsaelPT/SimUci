import React from "react";

const Dashboard: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h1 className="text-3xl font-semibold text-gray-900 dark:text-white mb-4">
          Welcome to CenterLab
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Your centralized laboratory management system
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Quick Stats
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            View your laboratory statistics here
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Recent Activity
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Check your recent laboratory activities
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Resources
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Access laboratory resources and documentation
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
