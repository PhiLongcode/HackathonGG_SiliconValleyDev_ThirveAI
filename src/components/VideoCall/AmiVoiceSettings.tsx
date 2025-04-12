import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button"; // Fixed import path

type AmiVoiceSettingsProps = {
  currentVoice: { gender: string; region: string };
  onVoiceChange: (voice: { gender: string; region: string }) => void;
};

export const  AmiVoiceSettings = ({ currentVoice, onVoiceChange }: AmiVoiceSettingsProps) => {
  return (
    <Dialog>
      <DialogTrigger asChild>
        {/* No changes needed here - button was already correct */}
        <Button variant="outline" size="icon" className="rounded-full w-12 h-12">
          <Settings className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Cài đặt giọng nói Ami</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Giới tính</Label>
            <Select
              value={currentVoice.gender}
              onValueChange={(value) => onVoiceChange({ ...currentVoice, gender: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Chọn giới tính" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="female">Nữ</SelectItem>
                <SelectItem value="male">Nam</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Vùng miền</Label>
            <Select
              value={currentVoice.region}
              onValueChange={(value) => onVoiceChange({ ...currentVoice, region: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Chọn vùng miền" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="north">Miền Bắc</SelectItem>
                <SelectItem value="central">Miền Trung</SelectItem>
                <SelectItem value="south">Miền Nam</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};