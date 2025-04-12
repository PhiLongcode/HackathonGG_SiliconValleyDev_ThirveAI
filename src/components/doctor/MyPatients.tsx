
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Filter } from "lucide-react";

const patients = [
  { id: 1, name: "Emma Johnson", age: 28, issue: "Anxiety", status: "Needs Attention", nextSession: "Apr 10, 2025", risk: "Medium" },
  { id: 2, name: "Michael Chen", age: 35, issue: "Depression", status: "Stable", nextSession: "Apr 12, 2025", risk: "Low" },
  { id: 3, name: "Sophia Rodriguez", age: 23, issue: "Stress", status: "Improving", nextSession: "Apr 11, 2025", risk: "Low" },
  { id: 4, name: "James Wilson", age: 42, issue: "PTSD", status: "Critical", nextSession: "Apr 9, 2025", risk: "High" },
  { id: 5, name: "Olivia Garcia", age: 31, issue: "Anxiety", status: "Stable", nextSession: "Apr 15, 2025", risk: "Medium" },
  { id: 6, name: "William Kim", age: 29, issue: "Depression", status: "Needs Attention", nextSession: "Apr 14, 2025", risk: "Medium" },
  { id: 7, name: "Ava Martinez", age: 26, issue: "Stress", status: "Stable", nextSession: "Apr 16, 2025", risk: "Low" },
];

export const MyPatients = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">My Patients</h2>
          <p className="text-muted-foreground">Manage and monitor your patient list.</p>
        </div>
        <Button>Add New Patient</Button>
      </div>
      
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="relative w-full sm:w-auto max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search patients..." className="pl-10" />
            </div>
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              <Select defaultValue="all">
                <SelectTrigger className="w-full sm:w-[140px]">
                  <SelectValue placeholder="Risk Level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Risks</SelectItem>
                  <SelectItem value="high">High Risk</SelectItem>
                  <SelectItem value="medium">Medium Risk</SelectItem>
                  <SelectItem value="low">Low Risk</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="all">
                <SelectTrigger className="w-full sm:w-[140px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="needs-attention">Needs Attention</SelectItem>
                  <SelectItem value="stable">Stable</SelectItem>
                  <SelectItem value="improving">Improving</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" className="w-full sm:w-auto">
                <Filter className="mr-2 h-4 w-4" />
                More Filters
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Patient</TableHead>
                <TableHead>Age</TableHead>
                <TableHead>Issue</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Risk Level</TableHead>
                <TableHead>Next Session</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {patients.map((patient) => (
                <TableRow key={patient.id}>
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <Avatar className="h-8 w-8">
                        <AvatarFallback>{patient.name.charAt(0)}</AvatarFallback>
                      </Avatar>
                      <span className="font-medium">{patient.name}</span>
                    </div>
                  </TableCell>
                  <TableCell>{patient.age}</TableCell>
                  <TableCell>{patient.issue}</TableCell>
                  <TableCell>
                    <Badge 
                      variant={
                        patient.status === "Critical" ? "destructive" : 
                        patient.status === "Needs Attention" ? "secondary" :
                        patient.status === "Improving" ? "default" : "outline"
                      }
                    >
                      {patient.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge 
                      variant={
                        patient.risk === "High" ? "destructive" : 
                        patient.risk === "Medium" ? "secondary" : "outline"
                      }
                    >
                      {patient.risk}
                    </Badge>
                  </TableCell>
                  <TableCell>{patient.nextSession}</TableCell>
                  <TableCell className="text-right space-x-2">
                    <Button variant="ghost" size="sm">View</Button>
                    <Button variant="ghost" size="sm">Schedule</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};
