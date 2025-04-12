
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer } from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const data = [
  { category: "Anxiety", count: 65, improvement: 45 },
  { category: "Depression", count: 40, improvement: 30 },
  { category: "Stress", count: 80, improvement: 60 },
  { category: "Sleep Issues", count: 55, improvement: 40 },
  { category: "Mood Disorders", count: 35, improvement: 25 },
];

export const AiInsights = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">AI Insights</h2>
        <p className="text-muted-foreground">AI-powered analytics and mental health trends.</p>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Mental Health Issues Distribution</CardTitle>
            <CardDescription>
              Breakdown of mental health categories across patients
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ChartContainer
              config={{
                count: { color: "#9F7AEA" },
                improvement: { color: "#68D391" }
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={data}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="category" tick={{ fill: 'var(--muted-foreground)' }} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" name="Patients" fill="#9F7AEA" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="improvement" name="Improved" fill="#68D391" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>AI Response Effectiveness</CardTitle>
            <CardDescription>
              Measurement of AI effectiveness in patient engagement
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground text-sm">
                AI effectiveness visualization coming soon
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>AI-driven Patient Insights</CardTitle>
          <CardDescription>
            Key trends and patterns detected by our AI system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm">
              Our AI has detected the following trends across the platform:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-sm">
              <li>65% of users report improved mood after 4 weeks of using the platform</li>
              <li>Morning check-ins show better engagement rates than evening ones</li>
              <li>Users responding to AI prompts show 40% better mental health outcomes</li>
              <li>Text-based therapy shows similar effectiveness to video sessions for mild cases</li>
              <li>Personalized exercise recommendations led to 35% better adherence rates</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
