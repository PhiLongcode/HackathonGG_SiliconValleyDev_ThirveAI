import AmiAvatar from "@/components/AmiAvatar";
import { cn } from "@/lib/utils";
import { User } from "lucide-react";

type VideoCallAreaProps = {
  callWith: "ami" | "expert" | null;
  isConnecting: boolean;
  isLowConnection: boolean;
  expertInfo: { name: string; specialty: string } | null;
  isCameraOn: boolean;
  localVideoRef: React.RefObject<HTMLVideoElement>;
  remoteVideoRef: React.RefObject<HTMLVideoElement>;
  isSpeaking: boolean;
};

export const VideoCallArea = ({
  callWith,
  isConnecting,
  isLowConnection,
  expertInfo,
  isCameraOn,
  localVideoRef,
  remoteVideoRef,
  isSpeaking
}: VideoCallAreaProps) => {
  return (
    <div className="relative flex-1 bg-black flex items-center justify-center">
      {/* Main video content */}
      {isConnecting ? (
        <div className="text-center text-white">
          <div className="spinner mb-4 mx-auto w-12 h-12 border-4 border-t-transparent border-white rounded-full animate-spin"></div>
          <p className="text-xl">
            Đang kết nối {callWith === "expert" ? "chuyên gia" : "Ami"}...
          </p>
        </div>
      ) : (
        <div className="w-full h-full relative">
          {/* Remote video/avatar */}
          {callWith === "ami" ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <AmiAvatar size="xl" mood="happy" />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
              {isCameraOn ? (
                <video
                  ref={remoteVideoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="text-center text-white">
                  <div className="w-32 h-32 rounded-full bg-primary flex items-center justify-center mx-auto">
                    <User className="w-16 h-16" />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Local video preview */}
          <div className={cn(
            "absolute bottom-4 right-4 w-32 h-24 bg-gray-800 rounded-lg border-2 overflow-hidden z-20",
            isSpeaking ? "border-primary-500 shadow-glow" : "border-white/20"
          )}>
            {isCameraOn ? (
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-white text-sm">Camera tắt</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};