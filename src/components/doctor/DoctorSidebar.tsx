
import React from "react";
import { Home, Users, Calendar, LineChart, Video } from "lucide-react";
import { cn } from "@/lib/utils";

interface DoctorSidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const DoctorSidebar = ({ activeTab, setActiveTab }: DoctorSidebarProps) => {
  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: Home },
    { id: "patients", label: "My Patients", icon: Users },
    { id: "appointments", label: "Appointments", icon: Calendar },
    { id: "ai-analytics", label: "AI Analytics", icon: LineChart },
    { id: "video-consultation", label: "Video Consultation", icon: Video },
  ];

  return (
    <aside className="w-64 h-screen bg-sidebar border-r border-border flex-shrink-0 hidden md:block">
      <div className="p-6 border-b border-border">
        <h1 className="text-2xl font-bold text-primary">ThriveAI Doctor</h1>
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
