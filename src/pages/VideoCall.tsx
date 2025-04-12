import { useState, useRef, useEffect } from "react";
import { Video, Users, User, Phone, PhoneOff, Mic, MicOff, Camera, CameraOff, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import AmiAvatar from "@/components/AmiAvatar";
import { useIsMobile } from "@/hooks/use-mobile";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Drawer, DrawerContent, DrawerTrigger } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import { useNavigate, useParams } from "react-router-dom";
import { Settings } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import axios from "axios";

type VoiceGender = 'male' | 'female';
type VoiceRegion = 'north' | 'central' | 'south';

const VideoCall = () => {
  const isMobile = useIsMobile();
  const navigate = useNavigate();
  const { callType } = useParams<{ callType: string }>();
  
  const [isInCall, setIsInCall] = useState(false);
  const [callWith, setCallWith] = useState<"ami" | "expert" | null>(null);
  const [isMicOn, setIsMicOn] = useState(true);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const [expertInfo, setExpertInfo] = useState<{ name: string; specialty: string } | null>(null);
  const [isLowConnection, setIsLowConnection] = useState(false);
  const [chatMessage, setChatMessage] = useState("");
  const [chatMessages, setChatMessages] = useState<{text: string, sender: "user" | "other", timestamp: Date}[]>([]);
  const chatRef = useRef<HTMLDivElement>(null);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  
  // WebRTC related states and refs
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  
  // Check if we have a call type from URL params
  useEffect(() => {
    if (callType === "ami" || callType === "expert") {
      handleStartCall(callType);
    }
  }, [callType]);

  // Initialize WebRTC when starting a call
  useEffect(() => {
    if (isInCall && !isConnecting) {
      initializeWebRTC();
    }
    
    return () => {
      // Clean up WebRTC resources when component unmounts or call ends
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
      }
    };
  }, [isInCall, isConnecting]);
  
  // Handle mic and camera toggle
  useEffect(() => {
    if (localStreamRef.current) {
      // Handle microphone toggle
      localStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = isMicOn;
      });
      
      // For camera toggle, we need a more robust approach
      if (isCameraOn) {
        // Check if we need to restart the camera
        const videoTracks = localStreamRef.current.getVideoTracks();
        
        // If no active video tracks, we need to get a new stream
        if (videoTracks.length === 0 || videoTracks[0].readyState === "ended" || !videoTracks[0].enabled) {
          // Stop any existing tracks first
          videoTracks.forEach(track => track.stop());
          
          // Get a fresh video stream
          navigator.mediaDevices.getUserMedia({ video: true })
            .then(newStream => {
              const newVideoTrack = newStream.getVideoTracks()[0];
              
              if (newVideoTrack) {
                // Replace track in peer connection if it exists
                if (peerConnectionRef.current) {
                  const senders = peerConnectionRef.current.getSenders();
                  const videoSender = senders.find(sender => 
                    sender.track && sender.track.kind === 'video'
                  );
                  
                  if (videoSender) {
                    videoSender.replaceTrack(newVideoTrack);
                  }
                }
                
                // Add the new track to our local stream
                if (localStreamRef.current) {
                  // Remove old tracks
                  const oldTracks = localStreamRef.current.getVideoTracks();
                  oldTracks.forEach(track => {
                    localStreamRef.current?.removeTrack(track);
                    track.stop();
                  });
                  
                  // Add new track
                  localStreamRef.current.addTrack(newVideoTrack);
                  
                  // Force update the video element
                  if (localVideoRef.current) {
                    localVideoRef.current.srcObject = null;
                    setTimeout(() => {
                      if (localVideoRef.current) {
                        localVideoRef.current.srcObject = localStreamRef.current;
                      }
                    }, 50);
                  }
                }
              }
            })
            .catch(error => {
              console.error("Error restarting camera:", error);
              toast({
                title: "Lỗi camera",
                description: "Không thể kết nối lại với camera",
                variant: "destructive",
              });
              setIsCameraOn(false);
            });
        } else {
          // Just enable existing tracks
          videoTracks.forEach(track => {
            track.enabled = true;
          });
        }
      } else {
        // When turning camera off, disable all video tracks
        const videoTracks = localStreamRef.current.getVideoTracks();
        videoTracks.forEach(track => {
          track.enabled = false;
        });
      }
    }
  }, [isMicOn, isCameraOn]);
  
  const initializeWebRTC = async () => {
    try {
      // Get user media (camera and microphone)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      
      localStreamRef.current = stream;
      
      // Display local video
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
      
      // Create RTCPeerConnection
      const configuration: RTCConfiguration = {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      };
      
      const peerConnection = new RTCPeerConnection(configuration);
      peerConnectionRef.current = peerConnection;
      
      // Add local stream tracks to peer connection
      stream.getTracks().forEach(track => {
        peerConnection.addTrack(track, stream);
      });
      
      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          // In a real app, you would send this candidate to the remote peer
          console.log("New ICE candidate:", event.candidate);
        }
      };
      
      // Handle connection state changes
      peerConnection.onconnectionstatechange = () => {
        console.log("Connection state:", peerConnection.connectionState);
        
        if (peerConnection.connectionState === 'disconnected' || 
            peerConnection.connectionState === 'failed') {
          toast({
            title: "Kết nối bị gián đoạn",
            description: "Đang thử kết nối lại...",
            variant: "destructive",
          });
          setIsLowConnection(true);
        } else if (peerConnection.connectionState === 'connected') {
          setIsLowConnection(false);
        }
      };
      
      // Handle incoming tracks (for remote video)
      peerConnection.ontrack = (event) => {
        if (remoteVideoRef.current && event.streams[0]) {
          remoteVideoRef.current.srcObject = event.streams[0];
        }
      };
      
      if (callWith === "ami") {
        // For Ami, we don't need a real WebRTC connection
        // Just simulate a connection
        console.log("Connected with Ami (simulated)");
      } else if (callWith === "expert") {
        // In a real app, you would:
        // 1. Create an offer
        // 2. Set local description
        // 3. Send the offer to the remote peer
        // 4. Wait for answer
        // 5. Set remote description
        
        // For demo purposes, we'll just create an offer
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        console.log("Created offer:", offer);
        
        // Simulate receiving an answer after a delay
        setTimeout(async () => {
          // This is just a simulation - in a real app, you would receive a real answer
          const simulatedAnswer = {
            type: 'answer',
            sdp: offer.sdp
          } as RTCSessionDescriptionInit;
          
          await peerConnection.setRemoteDescription(simulatedAnswer);
          console.log("Set remote description (simulated)");
        }, 1000);
      }
      
    } catch (error) {
      console.error("Error initializing WebRTC:", error);
      toast({
        title: "Lỗi kết nối",
        description: "Không thể kết nối camera hoặc microphone",
        variant: "destructive",
      });
    }
  };

  const initializeSpeechRecognition = () => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.lang = "vi-VN"; // Ngôn ngữ tiếng Việt
      recognition.interimResults = true;
      recognition.continuous = false;

      recognition.onstart = () => {
        setListening(true);
        toast({
          title: "Đang lắng nghe...",
          description: "Hãy nói điều bạn muốn chia sẻ với Ami",
        });
      };

      recognition.onend = () => {
        setListening(false);
        toast({
          title: "Dừng lắng nghe",
          description: "Nhận dạng giọng nói đã kết thúc",
        });
      };

      recognition.onresult = async (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');

        if (event.results[0].isFinal) {
          addChatMessage(transcript, "user");
          await sendToAI(transcript);
        }
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setListening(false);
        toast({
          title: "Lỗi nhận dạng giọng nói",
          description: "Vui lòng thử lại.",
          variant: "destructive",
        });
      };

      recognitionRef.current = recognition;
    } else {
      toast({
        title: "Không hỗ trợ",
        description: "Trình duyệt của bạn không hỗ trợ nhận dạng giọng nói.",
        variant: "destructive",
      });
    }
  };

  const toggleListening = () => {
    if (!recognitionRef.current) {
      initializeSpeechRecognition();
      return;
    }

    if (!listening) {
      recognitionRef.current.start();
    } else {
      recognitionRef.current.stop();
    }
  };
  
  const handleStartCall = (type: "ami" | "expert") => {
    // Update URL to reflect call type without triggering a page reload
    navigate(`/video-call/${type}`, { replace: true });
    
    setCallWith(type);
    setIsConnecting(true);
    
    // Create a new call record in history
    const callRecord = {
      id: Date.now().toString(),
      type: type,
      startTime: new Date(),
      endTime: null,
      messages: []
    };
    
    // Save to localStorage (you could use a more robust storage solution)
    const callHistory = JSON.parse(localStorage.getItem('callHistory') || '[]');
    callHistory.push(callRecord);
    localStorage.setItem('callHistory', JSON.stringify(callHistory));
    localStorage.setItem('currentCallId', callRecord.id);
    
    // Simulate connection delay and possibly low connection
    setTimeout(() => {
      setIsConnecting(false);
      setIsInCall(true);
      
      if (type === "expert") {
        setExpertInfo({
          name: "Bác sĩ Nguyễn Thị An",
          specialty: "Tâm lý học lâm sàng",
        });

        // Simulate low connection scenario (25% chance)
        if (Math.random() < 0.25) {
          setTimeout(() => {
            setIsLowConnection(true);
            toast({
              title: "Kết nối không ổn định",
              description: "Đang tối ưu kết nối...",
              variant: "destructive",
            });

            // Simulate connection improvement after 5 seconds
            setTimeout(() => {
              setIsLowConnection(false);
              toast({
                title: "Kết nối đã ổn định",
                description: "Bạn có thể tiếp tục cuộc gọi",
              });
            }, 5000);
          }, 10000);
        }
      }
    }, 2000);
  };
  
  const handleEndCall = () => {
    // Update call history with end time
    const currentCallId = localStorage.getItem('currentCallId');
    if (currentCallId) {
      const callHistory = JSON.parse(localStorage.getItem('callHistory') || '[]');
      const updatedHistory = callHistory.map((call: { id: string; messages?: { text: string; sender: "user" | "other"; timestamp: Date; }[]; }) => {
        if (call.id === currentCallId) {
          return {
            ...call,
            endTime: new Date(),
            messages: chatMessages.map(msg => ({
              text: msg.text,
              sender: msg.sender,
              timestamp: msg.timestamp
            }))
          };
        }
        return call;
      });
      
      localStorage.setItem('callHistory', JSON.stringify(updatedHistory));
      localStorage.removeItem('currentCallId');
    }
    
    // Stop all tracks
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }
    
    setIsInCall(false);
    setCallWith(null);
    setExpertInfo(null);
    setIsMicOn(true);
    setIsCameraOn(true);
    setIsLowConnection(false);
    setChatMessages([]);
    
    // Navigate back to the main video call page
    navigate('/video-call', { replace: true });
  };

  const handleSendMessage = () => {
    if (!chatMessage.trim()) return;
    
    const timestamp = new Date();
    
    // Add user message
    const newMessage = { 
      text: chatMessage, 
      sender: "user" as const,
      timestamp 
    };
    
    setChatMessages([...chatMessages, newMessage]);
    setChatMessage("");
    
    // Update call history with new message
    const currentCallId = localStorage.getItem('currentCallId');
    if (currentCallId) {
      const callHistory = JSON.parse(localStorage.getItem('callHistory') || '[]');
      const updatedHistory = callHistory.map((call: { id: string; messages?: { text: string; sender: "user" | "other"; timestamp: Date }[] }) => {
        if (call.id === currentCallId) {
          const updatedMessages = [...(call.messages || []), newMessage];
          return {
            ...call,
            messages: updatedMessages
          };
        }
        return call;
      });
      
      localStorage.setItem('callHistory', JSON.stringify(updatedHistory));
    }
    
    // Gửi tin nhắn đến AI
    if (callWith === "ami") {
      sendToAI(chatMessage);
    }

    // Simulate response (only for demo)
    setTimeout(() => {
      const response = callWith === "ami" 
        ? "Ami hiểu cảm xúc của bạn. Hãy chia sẻ thêm nhé." 
        : "Tôi hiểu điều bạn đang trải qua. Hãy thử hít thở sâu.";
      
      const responseMessage = {
        text: response, 
        sender: "other" as const,
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, responseMessage]);
      
      // Update call history with response message
      if (currentCallId) {
        const callHistory = JSON.parse(localStorage.getItem('callHistory') || '[]');
        const updatedHistory = callHistory.map((call: { id: string; messages?: { text: string; sender: "user" | "other"; timestamp: Date }[] }) => {
          if (call.id === currentCallId) {
            const updatedMessages = [...(call.messages || []), responseMessage];
            return {
              ...call,
              messages: updatedMessages
            };
          }
          return call;
        });
        
        localStorage.setItem('callHistory', JSON.stringify(updatedHistory));
      }
      
      // Auto scroll to bottom
      if (chatRef.current) {
        chatRef.current.scrollTop = chatRef.current.scrollHeight;
      }
    }, 1000);
  };

  // Hàm gửi tin nhắn đến AI và nhận phản hồi
  const sendToAI = async (message: string) => {
    try {
      toast({
        title: "Đang xử lý...",
        description: "Ami đang suy nghĩ về câu trả lời",
      });
      const res = await axios.post("http://localhost:8000/process_text", { message }); // Cấu hình URL backend tại đây
      if (res.data && res.data.reply) {
        const aiReply = res.data.reply;
        addChatMessage(aiReply, "other");
        speak(aiReply);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      console.error("AI request error:", err);
      if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
        toast({
          title: "Lỗi kết nối",
          description: "Không thể kết nối đến máy chủ AI. Vui lòng kiểm tra máy chủ đã chạy chưa.",
          variant: "destructive",
        });
      } else {
        toast({
          title: "Lỗi xử lý",
          description: "Không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
          variant: "destructive",
        });
      }
    }
  };

  // Hàm thêm tin nhắn vào giao diện chat
  const addChatMessage = (text: string, sender: "user" | "other") => {
    setChatMessages(prev => [
      ...prev,
      { text, sender, timestamp: new Date() }
    ]);
    setTimeout(() => {
      if (chatRef.current) {
        chatRef.current.scrollTop = chatRef.current.scrollHeight;
      }
    }, 100);
  };

  // Hàm chuyển văn bản thành giọng nói
  const speak = (text: string) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "vi-VN";
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    synth.cancel();
    synth.speak(utterance);
  };

  if (isInCall) {
    return (
      <div className="h-[calc(100vh-theme(spacing.16))] md:h-[calc(100vh-theme(spacing.12))] flex flex-col">
        {/* Video call area */}
        <div className="relative flex-1 bg-black flex items-center justify-center">
          {isConnecting ? (
            <div className="text-center text-white">
              <div className="spinner mb-4 mx-auto w-12 h-12 border-4 border-t-transparent border-white rounded-full animate-spin"></div>
              <p className="text-xl">Đang kết nối {callWith === "expert" ? "chuyên gia" : "Ami"}...</p>
            </div>
          ) : (
            <>
              {/* Main video (other person) */}
              <div className="w-full h-full relative">
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
                
                {/* Low connection overlay */}
                {isLowConnection && (
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
                    <div className="text-white text-center">
                      <div className="spinner mb-4 mx-auto w-12 h-12 border-4 border-t-transparent border-white rounded-full animate-spin"></div>
                      <p className="text-xl">Đang tối ưu kết nối...</p>
                      <p className="mt-2 text-sm">Vui lòng đợi trong giây lát</p>
                    </div>
                  </div>
                )}
                
                {/* Expert info */}
                {callWith === "expert" && expertInfo && (
                  <div className="absolute top-4 left-4 bg-black/50 p-2 rounded-lg text-white z-20">
                    <p className="font-medium">{expertInfo.name}</p>
                    <p className="text-xs">{expertInfo.specialty}</p>
                  </div>
                )}
                
                {/* Self view */}
                {isCameraOn ? (
                  <div className="absolute bottom-4 right-4 w-32 h-24 bg-gray-800 rounded-lg border border-white/20 overflow-hidden z-20">
                    <video
                      ref={localVideoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                  </div>
                ) : (
                  <div className="absolute bottom-4 right-4 w-32 h-24 bg-gray-800 rounded-lg border border-white/20 overflow-hidden flex items-center justify-center z-20">
                    <div className="text-white text-sm">Camera tắt</div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
        
        {/* Call controls */}
        <div className="h-20 bg-background flex items-center justify-center gap-4">
          {/* Nút bật/tắt mic */}
          {/* <Button
            variant="outline"
            size="icon"
            className={cn(
              "rounded-full w-12 h-12",
              !isMicOn && "bg-red-100 text-red-500 border-red-200"
            )}
            onClick={() => setIsMicOn(!isMicOn)}
          >
            {isMicOn ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
          </Button> */}

          {/* Nút bật/tắt camera */}
          <Button
            variant="outline"
            size="icon"
            className={cn(
              "rounded-full w-12 h-12",
              !isCameraOn && "bg-red-100 text-red-500 border-red-200"
            )}
            onClick={() => setIsCameraOn(!isCameraOn)}
          >
            {isCameraOn ? <Camera className="h-5 w-5" /> : <CameraOff className="h-5 w-5" />}
          </Button>

          {/* Nút nói */}
          <Button
            variant="outline"
            size="icon"
            className={cn(
              "rounded-full w-12 h-12",
              listening
                ? "bg-green-500 text-white border-green-600 animate-pulse"
                : "bg-gray-100 text-gray-500 border-gray-200"
            )}
            onClick={toggleListening}
            title={listening ? "Dừng lắng nghe" : "Bắt đầu lắng nghe"}
          >
            <Mic className="h-5 w-5" />
          </Button>

          {/* Nút kết thúc cuộc gọi */}
          <Button
            variant="destructive"
            size="icon"
            className="rounded-full w-14 h-14 bg-red-500 hover:bg-red-600"
            onClick={handleEndCall}
          >
            <PhoneOff className="h-6 w-6" />
          </Button>
        </div>
        
        {/* Confirmation dialog when user tries to navigate away */}
        <Dialog>
          <DialogTrigger asChild>
            <div className="hidden">Trigger</div>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Kết thúc cuộc gọi?</DialogTitle>
              <DialogDescription>
                Bạn có chắc chắn muốn kết thúc cuộc gọi này?
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline">Hủy</Button>
              <Button variant="destructive" onClick={handleEndCall}>Kết thúc</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }
  
  // Call selection screen
  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Gọi video</h1>
          <p className="text-muted-foreground">
            Kết nối trực tiếp với Ami hoặc chuyên gia tâm lý
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle>Gọi video với Ami</CardTitle>
            <CardDescription>
              Trò chuyện với người bạn AI của bạn qua video
            </CardDescription>
          </CardHeader>
          
          <CardContent className="pt-6 pb-2 flex justify-center">
            <AmiAvatar size="lg" mood="happy" />
          </CardContent>
          
          <CardFooter className="flex justify-center pt-6 pb-6">
            <Button 
              className="thrive-button" 
              onClick={() => navigate('/video-call/ami')}
            >
              <Video className="mr-2 h-4 w-4" /> Bắt đầu cuộc gọi
            </Button>
          </CardFooter>
        </Card>
        
        <Card className="overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle>Gọi video với chuyên gia</CardTitle>
            <CardDescription>
              Kết nối với chuyên gia tâm lý qua video call
            </CardDescription>
          </CardHeader>
          
          <CardContent className="pt-6 pb-2 flex justify-center">
            <div className="w-24 h-24 rounded-full bg-primary flex items-center justify-center">
              <Users className="w-12 h-12 text-white" />
            </div>
          </CardContent>
          
          <CardFooter className="flex justify-center pt-6 pb-6">
            <Dialog>
              <DialogTrigger asChild>
                <Button className="thrive-button">
                  <Phone className="mr-2 h-4 w-4" /> Liên hệ chuyên gia
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Xác nhận cuộc gọi với chuyên gia</DialogTitle>
                  <DialogDescription>
                    Bạn sẽ được kết nối với chuyên gia tâm lý phù hợp để được hỗ trợ.
                  </DialogDescription>
                </DialogHeader>
                
                <div className="py-4">
                  <p className="text-muted-foreground">
                    Lưu ý: Cuộc gọi sẽ được bảo mật. Thông tin của bạn sẽ chỉ được chia sẻ với chuyên gia để hỗ trợ tốt nhất.
                  </p>
                </div>
                
                <DialogFooter>
                  <Button variant="outline">Hủy</Button>
                  <Button onClick={() => handleStartCall("expert")}>Tiếp tục</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </CardFooter>
        </Card>
      </div>
      
      <div className="thrive-card p-6 border border-thrive-lavender">
        <h3 className="font-medium text-lg mb-2">Thông tin về cuộc gọi</h3>
        <ul className="space-y-2 text-muted-foreground">
          <li>• Cuộc gọi với Ami hoàn toàn miễn phí và không giới hạn thời gian.</li>
          <li>• Cuộc gọi với chuyên gia sẽ được kết nối với bác sĩ tâm lý chuyên nghiệp.</li>
          <li>• Mọi cuộc trò chuyện đều được bảo mật và tuân thủ quy định về riêng tư.</li>
          <li>• Lịch sử cuộc gọi được lưu trữ để bạn có thể xem lại sau này.</li>
        </ul>
      </div>
    </div>
  );
};

export default VideoCall;