
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

const users = [
  { id: 1, name: "Dr. Sarah Johnson", email: "sarah.j@thrive.ai", role: "doctor", status: "active" },
  { id: 2, name: "Michael Chen", email: "m.chen@thrive.ai", role: "patient", status: "active" },
  { id: 3, name: "Dr. Robert Williams", email: "robert.w@thrive.ai", role: "doctor", status: "inactive" },
  { id: 4, name: "Emily Davis", email: "emily.d@thrive.ai", role: "patient", status: "active" },
  { id: 5, name: "Jessica Kim", email: "jessica.k@thrive.ai", role: "patient", status: "active" },
  { id: 6, name: "Dr. James Wilson", email: "james.w@thrive.ai", role: "doctor", status: "active" },
];

export const UserManagement = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">User Management</h2>
          <p className="text-muted-foreground">Manage and monitor all your users from here.</p>
        </div>
        <Button>Add New User</Button>
      </div>
      
      <Card>
        <CardHeader className="pb-3">
          <Tabs defaultValue="all">
            <TabsList>
              <TabsTrigger value="all">All Users</TabsTrigger>
              <TabsTrigger value="doctors">Doctors</TabsTrigger>
              <TabsTrigger value="patients">Patients</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 mb-6">
            <Input placeholder="Search users..." className="max-w-sm" />
            <Select defaultValue="all">
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Email</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell className="font-medium">{user.name}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>
                    <Badge variant={user.role === "doctor" ? "secondary" : "outline"}>
                      {user.role === "doctor" ? "Doctor" : "Patient"}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={user.status === "active" ? "default" : "destructive"}>
                      {user.status === "active" ? "Active" : "Inactive"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right space-x-2">
                    <Button variant="ghost" size="sm">Edit</Button>
                    <Button variant="ghost" size="sm">Delete</Button>
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
