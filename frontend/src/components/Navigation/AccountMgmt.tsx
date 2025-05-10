import React, { useContext } from "react";
import { SidebarContext } from "./Sidebar";

const DEFAULT_PFP = "pictures/avatars/uifaces-abstract-image_1.jpg";

type AccountMgmtProps = {
  userName: string;
  email: string;
  profilePicture?: string;
};

const AccountMgmt: React.FC<AccountMgmtProps> = ({
  userName,
  email,
  profilePicture = DEFAULT_PFP,
}) => {
  const { expanded } = useContext(SidebarContext);

  return (
    <>
      <div
        className={`flex rounded-3xl justify-center-safe ${
          expanded
            ? "border-2 border-emerald-900 bg-white p-2 m-1"
            : "border-0 bg-transparent p-1 m-0"
        }`}
      >
        <div
          className={`${
            expanded ? "mr-2" : "mr-0"
          } flex items-center justify-center`}
        >
          <img
            src={profilePicture}
            alt={`${userName}'s profile`}
            className={`${
              expanded ? "size-16" : "size-8"
            } rounded-full object-cover border-2 border-emerald-900`}
            onError={(e) => {
              (e.target as HTMLImageElement).src = DEFAULT_PFP;
            }}
          />
        </div>
        <div className="flex flex-col justify-center">
          {expanded && (
            <div>
              <h2 className="text-sm font-semibold text-gray-800">
                {userName}
              </h2>
              <p className="text-xs italic text-gray-600">{email}</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default AccountMgmt;
