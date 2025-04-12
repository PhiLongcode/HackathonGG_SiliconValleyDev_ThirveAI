
import React from "react";
import { Home, Users, FileText, LineChart, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

interface AdminSidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const AdminSidebar = ({ activeTab, setActiveTab }: AdminSidebarProps) => {
  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: Home },
    { id: "users", label: "Users", icon: Users },
    { id: "reports", label: "Reports", icon: FileText },
    { id: "ai-insights", label: "AI Insights", icon: LineChart },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside className="w-64 h-screen bg-sidebar border-r border-border flex-shrink-0 hidden md:block">
      <div className="p-6 border-b border-border">
        <h1 className="text-2xl font-bold text-primary">ThriveAI Admin</h1>
      </div>
      
      <nav className="p-4">
        <ul className="space-y-2">
          {menuItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-4 py-3 rounded-md text-sm font-medium transition-colors",
                  activeTab === item.id
                    ? "bg-primary/10 text-primary"
                    : "hover:bg-muted text-foreground/70 hover:text-foreground"
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
};
