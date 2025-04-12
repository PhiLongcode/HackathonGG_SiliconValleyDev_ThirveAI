
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Calendar, Clock, Users, Video } from "lucide-react";
import { ChartContainer } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const moodData = [
  { day: "Mon", anxiety: 4, depression: 3, sleep: 5 },
  { day: "Tue", anxiety: 3, depression: 3, sleep: 4 },
  { day: "Wed", anxiety: 5, depression: 4, sleep: 3 },
  { day: "Thu", anxiety: 3, depression: 2, sleep: 4 },
  { day: "Fri", anxiety: 2, depression: 2, sleep: 5 },
  { day: "Sat", anxiety: 1, depression: 1, sleep: 6 },
  { day: "Sun", anxiety: 2, depression: 2, sleep: 5 },
];

const recentPatients = [
  { id: 1, name: "Emma Johnson", status: "Needs Attention", avatar: "", lastSession: "April 3, 2025", issue: "Anxiety" },
  { id: 2, name: "Michael Chen", status: "Stable", avatar: "", lastSession: "April 5, 2025", issue: "Depression" },
  { id: 3, name: "Sophia Rodriguez", status: "Improving", avatar: "", lastSession: "April 6, 2025", issue: "Stress" },
  { id: 4, name: "James Wilson", status: "Critical", avatar: "", lastSession: "April 2, 2025", issue: "PTSD" },
];

const upcomingAppointments = [
  { id: 1, patient: "Emma Johnson", time: "10:00 AM", duration: "45 min", type: "Video Call" },
  { id: 2, patient: "David Kim", time: "11:30 AM", duration: "30 min", type: "Chat Session" },
  { id: 3, patient: "Sophia Rodriguez", time: "2:15 PM", duration: "45 min", type: "Video Call" },
];

const statsItems = [
  {
    title: "Active Patients",
    value: "28",
    description: "+3 this week",
    icon: Users,
    color: "bg-thrive-blue",
  },
  {
    title: "Today's Sessions",
    value: "8",
    description: "2 pending",
    icon: Calendar,
    color: "bg-thrive-lavender",
  },
  {
    title: "Total Hours",
    value: "32",
    description: "This week",
    icon: Clock,
    color: "bg-thrive-teal",
  },
  {
    title: "Video Calls",
    value: "15",
    description: "Last 7 days",
    icon: Video,
    color: "bg-thrive-peach",
  },
];

export const DoctorDashboard = () => {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Doctor Dashboard</h2>
      <p className="text-muted-foreground">Welcome back, Dr. Smith. Here's an overview of your practice.</p>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {statsItems.map((item, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">{item.title}</CardTitle>
              <div className={`${item.color} p-2 rounded-md`}>
                <item.icon className="h-4 w-4 text-background" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{item.value}</div>
              <p className="text-xs text-muted-foreground">{item.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Patient Mood Trends</CardTitle>
            <CardDescription>
              Average self-reported metrics for the past week
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ChartContainer
              config={{
                anxiety: { color: "#E5DEFF" },
                depression: { color: "#D3E4FD" },
                sleep: { color: "#D3F2EA" }
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={moodData}
                  margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="day" tick={{ fill: 'var(--muted-foreground)' }} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)' }} domain={[0, 10]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="anxiety" name="Anxiety" stroke="#9F7AEA" strokeWidth={2} activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="depression" name="Depression" stroke="#4299E1" strokeWidth={2} />
                  <Line type="monotone" dataKey="sleep" name="Sleep Quality" stroke="#68D391" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
        
        <Card className="col-span-1">
          <CardHeader className="flex justify-between items-center">
            <div>
              <CardTitle>Today's Appointments</CardTitle>
              <CardDescription>
                Your upcoming sessions for today
              </CardDescription>
            </div>
            <Button variant="outline" size="sm">View All</Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {upcomingAppointments.map((appointment) => (
                <div key={appointment.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="bg-primary/10 p-2 rounded-full">
                      <Clock className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{appointment.patient}</p>
                      <p className="text-xs text-muted-foreground">{appointment.time} • {appointment.duration}</p>
                    </div>
                  </div>
                  <Badge variant="outline">{appointment.type}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader className="flex justify-between items-center">
          <div>
            <CardTitle>Recent Patients</CardTitle>
            <CardDescription>
              Overview of your recently active patients
            </CardDescription>
          </div>
          <Button variant="outline" size="sm">View All</Button>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {recentPatients.map((patient) => (
              <Card key={patient.id} className="overflow-hidden">
                <CardHeader className="p-4">
                  <div className="flex items-center gap-3">
                    <Avatar>
                      <AvatarImage src={patient.avatar} />
                      <AvatarFallback>{patient.name.charAt(0)}</AvatarFallback>
                    </Avatar>
                    <div>
                      <CardTitle className="text-sm">{patient.name}</CardTitle>
                      <CardDescription className="text-xs">
                        Last session: {patient.lastSession}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">{patient.issue}</Badge>
                    <Badge 
                      variant={
                        patient.status === "Critical" ? "destructive" : 
                        patient.status === "Needs Attention" ? "secondary" :
                        patient.status === "Improving" ? "default" : "outline"
                      }
                      className="text-xs"
                    >
                      {patient.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
