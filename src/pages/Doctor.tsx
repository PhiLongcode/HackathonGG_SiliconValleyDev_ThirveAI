
import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { DoctorSidebar } from "@/components/doctor/DoctorSidebar";
import { DoctorHeader } from "@/components/doctor/DoctorHeader";
import { DoctorDashboard } from "@/components/doctor/DoctorDashboard";
import { MyPatients } from "@/components/doctor/MyPatients";
import { Appointments } from "@/components/doctor/Appointments";
import { DoctorAiAnalytics } from "@/components/doctor/DoctorAiAnalytics";
import { VideoConsultation } from "@/components/doctor/VideoConsultation";

const Doctor = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  
  return (
    <div className="flex h-screen bg-background">
      <DoctorSidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="flex flex-col flex-1 overflow-hidden">
        <DoctorHeader />
        
        <main className="flex-1 overflow-y-auto p-6">
          {activeTab === "dashboard" && <DoctorDashboard />}
          {activeTab === "patients" && <MyPatients />}
          {activeTab === "appointments" && <Appointments />}
          {activeTab === "ai-analytics" && <DoctorAiAnalytics />}
          {activeTab === "video-consultation" && <VideoConsultation />}
        </main>
      </div>
    </div>
  );
};

export default Doctor;
