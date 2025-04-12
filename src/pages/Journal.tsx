
import { useState } from "react";
import { Plus, Calendar, Edit, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface JournalEntry {
  id: string;
  title: string;
  content: string;
  date: Date;
  isPublic: boolean;
}

const mockEntries: JournalEntry[] = [
  {
    id: "1",
    title: "Ngày làm việc bận rộn",
    content: "Hôm nay thật sự rất bận rộn nhưng mình đã hoàn thành được nhiều việc. Cảm thấy hơi mệt nhưng vẫn khá hài lòng với những gì đã làm được.",
    date: new Date("2025-04-05"),
    isPublic: false,
  },
  {
    id: "2",
    title: "Cuộc gặp với bạn bè",
    content: "Mình đã gặp lại những người bạn cũ sau một thời gian dài. Thật vui khi được trò chuyện và chia sẻ những câu chuyện. Cảm thấy tinh thần phấn chấn hơn rất nhiều.",
    date: new Date("2025-04-04"),
    isPublic: true,
  },
  {
    id: "3",
    title: "Ngày không tốt",
    content: "Hôm nay thực sự không phải là một ngày tốt. Mình gặp nhiều khó khăn và cảm thấy khá căng thẳng. Mong rằng ngày mai sẽ tốt hơn.",
    date: new Date("2025-04-03"),
    isPublic: false,
  },
];

const Journal = () => {
  const [entries, setEntries] = useState<JournalEntry[]>(mockEntries);
  const [newEntry, setNewEntry] = useState<Omit<JournalEntry, "id">>({
    title: "",
    content: "",
    date: new Date(),
    isPublic: false,
  });
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  
  const handleCreateEntry = () => {
    if (!newEntry.title || !newEntry.content) return;
    
    const entry: JournalEntry = {
      ...newEntry,
      id: Date.now().toString(),
    };
    
    setEntries([entry, ...entries]);
    setNewEntry({
      title: "",
      content: "",
      date: new Date(),
      isPublic: false,
    });
    
    setIsDialogOpen(false);
  };
  
  const handleDeleteEntry = (id: string) => {
    setEntries(entries.filter(entry => entry.id !== id));
  };
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Nhật ký cảm xúc</h1>
          <p className="text-muted-foreground">
            Ghi lại cảm xúc và những suy nghĩ của bạn
          </p>
        </div>
        
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button className="thrive-button">
              <Plus className="mr-2 h-4 w-4" /> Tạo nhật ký mới
            </Button>
          </DialogTrigger>
          
          <DialogContent className="sm:max-w-[500px]">
            <DialogHeader>
              <DialogTitle>Tạo nhật ký mới</DialogTitle>
              <DialogDescription>
                Hãy chia sẻ cảm xúc và suy nghĩ của bạn ngày hôm nay
              </DialogDescription>
            </DialogHeader>
            
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="entry-title" className="text-right">
                  Tiêu đề
                </Label>
                <Input
                  id="entry-title"
                  placeholder="Nhập tiêu đề"
                  className="col-span-3"
                  value={newEntry.title}
                  onChange={(e) => setNewEntry({ ...newEntry, title: e.target.value })}
                />
              </div>
              
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="entry-content" className="text-right">
                  Nội dung
                </Label>
                <Textarea
                  id="entry-content"
                  placeholder="Hãy viết cảm xúc của bạn..."
                  className="col-span-3 min-h-[150px]"
                  value={newEntry.content}
                  onChange={(e) => setNewEntry({ ...newEntry, content: e.target.value })}
                />
              </div>
              
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="entry-public" className="text-right">
                  Công khai
                </Label>
                <div className="flex items-center space-x-2 col-span-3">
                  <Switch
                    id="entry-public"
                    checked={newEntry.isPublic}
                    onCheckedChange={(checked) => setNewEntry({ ...newEntry, isPublic: checked })}
                  />
                  <Label htmlFor="entry-public">
                    Chia sẻ với cộng đồng (ẩn danh)
                  </Label>
                </div>
              </div>
            </div>
            
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Hủy
              </Button>
              <Button onClick={handleCreateEntry}>Lưu nhật ký</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {entries.map((entry) => (
          <Card key={entry.id} className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <CardTitle>{entry.title}</CardTitle>
                <div className="flex gap-2">
                  <Button variant="ghost" size="icon">
                    <Edit className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="icon" onClick={() => handleDeleteEntry(entry.id)}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="pb-2">
              <p className="text-muted-foreground line-clamp-3">{entry.content}</p>
            </CardContent>
            
            <CardFooter className="flex justify-between pt-2 text-xs text-muted-foreground">
              <div className="flex items-center">
                <Calendar className="h-3 w-3 mr-1" />
                {entry.date.toLocaleDateString("vi-VN")}
              </div>
              
              {entry.isPublic ? (
                <span className="text-primary text-xs">Công khai</span>
              ) : (
                <span className="text-muted-foreground text-xs">Riêng tư</span>
              )}
            </CardFooter>
          </Card>
        ))}
      </div>
      
      {entries.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">
            Bạn chưa có nhật ký nào. Hãy tạo nhật ký đầu tiên!
          </p>
        </div>
      )}
    </div>
  );
};

export default Journal;
