import React, {
  createContext,
  useContext,
  useState,
  PropsWithChildren,
} from "react";
import { Link } from "react-router-dom";
import {
  LuArrowLeftToLine,
  LuArrowRightToLine,
  LuSun,
  LuMoon,
} from "react-icons/lu";
import { useTheme } from "../../context/ThemeContext";
import AccountMgmt from "./AccountMgmt";

export const SidebarContext = createContext({ expanded: true });

export const SidebarItem: React.FC<{
  text: string;
  href: string;
  icon?: React.ReactNode;
}> = ({ text = "Item", href = "/", icon }) => {
  const { expanded } = useContext(SidebarContext);

  return (
    <li className="relative group">
      <Link
        to={href}
        className="flex items-center p-2 hover:bg-emerald-200 dark:hover:bg-emerald-800 rounded-lg transition-colors"
      >
        <span className="inline-flex items-center justify-center">{icon}</span>
        <span
          className={`overflow-hidden transition-all ${
            expanded ? "w-52 ml-3" : "w-0"
          }`}
        >
          {text}
        </span>
        {!expanded && (
          <div className="absolute left-full rounded-md px-2 py-1 ml-6 bg-emerald-100 dark:bg-emerald-800 text-emerald-800 dark:text-emerald-100 text-sm invisible opacity-20 -translate-x-3 transition-all group-hover:visible group-hover:opacity-100 group-hover:translate-x-0">
            {text}
          </div>
        )}
      </Link>
    </li>
  );
};

const Sidebar: React.FC<PropsWithChildren> = ({ children }) => {
  const [expanded, setExpanded] = useState(true);
  const { darkMode, toggleDarkMode } = useTheme();

  return (
    <aside className="h-screen sticky top-0">
      <nav className="h-full flex flex-col bg-gradient-to-b from-emerald-50 via-emerald-200 to-emerald-50 dark:bg-gray-900 border-r shadow-sm">
        <div className="p-4 pb-2 flex justify-end">
          <button
            onClick={() => setExpanded((curr) => !curr)}
            className="p-2 rounded-lg border-2 border-emerald-200 bg-emerald-100 hover:bg-emerald-50 transition-all dark:bg-emerald-800 dark:hover:bg-emerald-700"
          >
            {expanded ? <LuArrowLeftToLine /> : <LuArrowRightToLine />}
          </button>
        </div>

        <SidebarContext.Provider value={{ expanded }}>
          <ul className="flex-1 px-3">{children}</ul>

          <div className="border-t items-center justify-center transition-all dark:border-gray-800 p-3">
            <AccountMgmt
              userName="Alberto De Las Mercedes"
              email="alberto@email.com"
            />
            <button
              onClick={toggleDarkMode}
              className="flex items-center justify-center w-full p-2 rounded-lg bg-emerald-50 hover:bg-emerald-100 dark:bg-emerald-800 dark:hover:bg-emerald-700 transition-colors"
            >
              {darkMode ? (
                <div className="flex items-center gap-2">
                  <LuSun className="text-amber-500" size={20} />
                  {expanded && <span>Light Mode</span>}
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <LuMoon
                    className="text-slate-700 dark:text-slate-200"
                    size={20}
                  />
                  {expanded && <span>Dark Mode</span>}
                </div>
              )}
            </button>
          </div>
        </SidebarContext.Provider>
      </nav>
    </aside>
  );
};

export default Sidebar;
