
import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AdminSidebar } from "@/components/admin/AdminSidebar";
import { AdminHeader } from "@/components/admin/AdminHeader";
import { DashboardOverview } from "@/components/admin/DashboardOverview";
import { UserManagement } from "@/components/admin/UserManagement";
import { AiInsights } from "@/components/admin/AiInsights";
import { Reports } from "@/components/admin/Reports";
import { Settings } from "@/components/admin/Settings";

const Admin = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  
  return (
    <div className="flex h-screen bg-background">
      <AdminSidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="flex flex-col flex-1 overflow-hidden">
        <AdminHeader />
        
        <main className="flex-1 overflow-y-auto p-6">
          {activeTab === "dashboard" && <DashboardOverview />}
          {activeTab === "users" && <UserManagement />}
          {activeTab === "ai-insights" && <AiInsights />}
          {activeTab === "reports" && <Reports />}
          {activeTab === "settings" && <Settings />}
        </main>
      </div>
    </div>
  );
};

export default Admin;
