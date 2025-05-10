import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./components/DashboardApp";
import Sidebar, { SidebarItem } from "./components/Navigation/Sidebar";
import DashboardSimulation from "./components/Simulation/DashboardSimulation";
import { LuLayoutDashboard, LuBeaker } from "react-icons/lu";

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-gray-50 dark:bg-gray-900">
        <Sidebar>
          <SidebarItem
            icon={<LuLayoutDashboard size={20} />}
            text="Home"
            href="/"
          />
          <SidebarItem
            icon={<LuBeaker size={20} />}
            text="Simulation"
            href="/simulation"
          />
        </Sidebar>
        <main className="flex-1 p-4 transition-all duration-200 text-gray-900 dark:text-gray-100">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulation" element={<DashboardSimulation />} />
          </Routes>

          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
            <p>
              Lorem ipsum dolor sit, amet consectetur adipisicing elit.
              Accusantium provident quod vitae placeat doloremque harum
              reiciendis voluptatum enim beatae ipsam! Quod architecto optio
              atque quasi molestias reprehenderit. Quasi, asperiores vitae.
            </p>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
};

export default App;
