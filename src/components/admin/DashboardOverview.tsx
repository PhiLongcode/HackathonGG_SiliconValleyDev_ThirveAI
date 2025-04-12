
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { UserPlus, Users, Video, FileText } from "lucide-react";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { month: "Jan", patients: 400, doctors: 50, consultations: 240 },
  { month: "Feb", patients: 500, doctors: 55, consultations: 280 },
  { month: "Mar", patients: 600, doctors: 60, consultations: 320 },
  { month: "Apr", patients: 620, doctors: 65, consultations: 350 },
  { month: "May", patients: 700, doctors: 70, consultations: 410 },
  { month: "Jun", patients: 800, doctors: 80, consultations: 450 },
];

const statsItems = [
  {
    title: "Total Doctors",
    value: "80",
    description: "+12% from last month",
    icon: UserPlus,
    color: "bg-thrive-blue",
  },
  {
    title: "Total Patients",
    value: "800",
    description: "+25% from last month",
    icon: Users,
    color: "bg-thrive-lavender",
  },
  {
    title: "Consultations",
    value: "450",
    description: "+15% from last month",
    icon: Video,
    color: "bg-thrive-teal",
  },
  {
    title: "Mental Health Reports",
    value: "210",
    description: "+8% from last month",
    icon: FileText,
    color: "bg-thrive-peach",
  },
];

export const DashboardOverview = () => {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
      <p className="text-muted-foreground">Welcome to the ThriveAI admin dashboard.</p>
      
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
            <CardTitle>Platform Growth</CardTitle>
            <CardDescription>
              Users and consultations over the last 6 months
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ChartContainer
              config={{
                patients: { color: "#9F7AEA" },
                doctors: { color: "#68D391" },
                consultations: { color: "#4299E1" }
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={data}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="month" tick={{ fill: 'var(--muted-foreground)' }} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)' }} />
                  <ChartTooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <ChartTooltipContent>
                            <div className="font-semibold">{label}</div>
                            {payload.map((entry, index) => (
                              <div key={`item-${index}`} className="flex gap-2 items-center">
                                <div
                                  className="h-2 w-2 rounded-full"
                                  style={{ backgroundColor: entry.color }}
                                />
                                <span>{entry.name}: {entry.value}</span>
                              </div>
                            ))}
                          </ChartTooltipContent>
                        );
                      }
                      return null;
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="patients"
                    stackId="1"
                    stroke="#9F7AEA"
                    fill="#9F7AEA"
                    fillOpacity={0.2}
                  />
                  <Area
                    type="monotone"
                    dataKey="doctors"
                    stackId="1"
                    stroke="#68D391"
                    fill="#68D391"
                    fillOpacity={0.2}
                  />
                  <Area
                    type="monotone"
                    dataKey="consultations"
                    stackId="1"
                    stroke="#4299E1"
                    fill="#4299E1"
                    fillOpacity={0.2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
        
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Mental Health Insights</CardTitle>
            <CardDescription>
              AI analysis of user mental health patterns
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground text-sm">
                Detailed mental health analytics visualization coming soon
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
