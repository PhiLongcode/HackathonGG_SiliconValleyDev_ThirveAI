
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileDown, FileText, Filter } from "lucide-react";

const reports = [
  { id: 1, name: "Monthly User Growth Report", date: "Apr 5, 2025", type: "system", status: "ready" },
  { id: 2, name: "Q1 Mental Health Trends", date: "Apr 2, 2025", type: "analytics", status: "ready" },
  { id: 3, name: "User Engagement Analysis", date: "Mar 28, 2025", type: "analytics", status: "processing" },
  { id: 4, name: "AI Response Effectiveness", date: "Mar 25, 2025", type: "analytics", status: "ready" },
  { id: 5, name: "Therapist Performance Review", date: "Mar 20, 2025", type: "system", status: "ready" },
  { id: 6, name: "Patient Improvement Metrics", date: "Mar 15, 2025", type: "analytics", status: "ready" },
];

export const Reports = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Reports</h2>
          <p className="text-muted-foreground">View and download system reports and analytics.</p>
        </div>
        <Button>
          <FileText className="mr-2 h-4 w-4" />
          Generate New Report
        </Button>
      </div>
      
      <Card>
        <CardHeader className="pb-3">
          <Tabs defaultValue="all">
            <TabsList>
              <TabsTrigger value="all">All Reports</TabsTrigger>
              <TabsTrigger value="system">System Reports</TabsTrigger>
              <TabsTrigger value="analytics">Analytics Reports</TabsTrigger>
            </TabsList>
          
            <TabsContent value="all">
              <div className="flex justify-between my-6">
                <Button variant="outline" size="sm">
                  <Filter className="mr-2 h-4 w-4" />
                  Filter
                </Button>
                <div className="space-x-2">
                  <Button variant="outline" size="sm">Export All</Button>
                </div>
              </div>
              
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Report Name</TableHead>
                    <TableHead>Date Created</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {reports.map((report) => (
                    <TableRow key={report.id}>
                      <TableCell className="font-medium">{report.name}</TableCell>
                      <TableCell>{report.date}</TableCell>
                      <TableCell className="capitalize">{report.type}</TableCell>
                      <TableCell className="capitalize">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          report.status === "ready" 
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300" 
                            : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300"
                        }`}>
                          {report.status}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button 
                          variant="ghost" 
                          size="sm"
                          disabled={report.status !== "ready"}
                        >
                          <FileDown className="mr-2 h-4 w-4" />
                          Download
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
            
            <TabsContent value="system">
              <div className="p-4 text-center text-muted-foreground">
                Showing only system reports
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Report Name</TableHead>
                    <TableHead>Date Created</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {reports.filter(report => report.type === 'system').map((report) => (
                    <TableRow key={report.id}>
                      <TableCell className="font-medium">{report.name}</TableCell>
                      <TableCell>{report.date}</TableCell>
                      <TableCell className="capitalize">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          report.status === "ready" 
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300" 
                            : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300"
                        }`}>
                          {report.status}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button 
                          variant="ghost" 
                          size="sm"
                          disabled={report.status !== "ready"}
                        >
                          <FileDown className="mr-2 h-4 w-4" />
                          Download
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
            
            <TabsContent value="analytics">
              <div className="p-4 text-center text-muted-foreground">
                Showing only analytics reports
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Report Name</TableHead>
                    <TableHead>Date Created</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {reports.filter(report => report.type === 'analytics').map((report) => (
                    <TableRow key={report.id}>
                      <TableCell className="font-medium">{report.name}</TableCell>
                      <TableCell>{report.date}</TableCell>
                      <TableCell className="capitalize">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          report.status === "ready" 
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300" 
                            : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300"
                        }`}>
                          {report.status}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button 
                          variant="ghost" 
                          size="sm"
                          disabled={report.status !== "ready"}
                        >
                          <FileDown className="mr-2 h-4 w-4" />
                          Download
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
          </Tabs>
        </CardHeader>
      </Card>
    </div>
  );
};
