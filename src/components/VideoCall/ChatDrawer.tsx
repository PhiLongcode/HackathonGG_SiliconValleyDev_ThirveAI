import { Drawer, DrawerContent, DrawerTrigger } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button"; // Fixed import
import { cn } from "@/lib/utils"; // Added missing import

type ChatDrawerProps = {
  chatRef: React.RefObject<HTMLDivElement>;
  chatMessages: { text: string; sender: "user" | "other"; timestamp: Date }[];
  chatMessage: string;
  setChatMessage: (message: string) => void;
  onSendMessage: () => void;
};

export const ChatDrawer = ({
  chatRef,
  chatMessages,
  chatMessage,
  setChatMessage,
  onSendMessage
}: ChatDrawerProps) => {
  return (
    <Drawer>
      <DrawerTrigger asChild>
        <Button variant="outline" size="icon" className="rounded-full w-12 h-12">
          <MessageCircle className="h-5 w-5" />
        </Button>
      </DrawerTrigger>
      <DrawerContent className="h-[70vh]">
        <div className="p-4 flex flex-col h-full">
          <h2 className="text-lg font-medium mb-2">Tin nhắn</h2>
          <div ref={chatRef} className="flex-1 overflow-y-auto p-2 space-y-2 mb-2">
            {chatMessages.map((msg, idx) => (
              <div
                key={idx}
                className={cn(
                  "max-w-[80%] rounded-3xl p-3",
                  msg.sender === "user"
                    ? "bg-primary text-primary-foreground ml-auto rounded-tr-none"
                    : "bg-muted mr-auto rounded-tl-none"
                )}
              >
                {msg.text}
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Nhập tin nhắn..."
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && onSendMessage()}
              className="flex-1"
            />
            <Button onClick={onSendMessage}>Gửi</Button>
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
};