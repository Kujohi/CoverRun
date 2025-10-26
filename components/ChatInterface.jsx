"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Send, Bot, User, Trash2, Image, X } from 'lucide-react';
import { supabase } from '@/services/supabaseClient';
import { useDataRefresh } from '@/context/DataRefreshContext';

const SUGGESTIONS = [
  {
    label: 'Tôi đã chi bao nhiêu cho cà phê/thức uống tuần này?',
    value: 'Tôi đã chi bao nhiêu cho cà phê hoặc nước uống tuần này?'
  },
  {
    label: 'Khi nào tôi có thể tiết kiệm được 50 triệu?',
    value: 'Khi nào tôi có thể tiết kiệm được 50 triệu?'
  },
  {
    label: 'Tôi vừa chi 5 triệu để trả tiền thuê nhà.',
    value: 'Tôi vừa chi 5 triệu để trả tiền thuê nhà.'
  },
  {
    label: 'Tôi muốn đặt mục tiêu tiết kiệm 50 triệu',
    value: 'Tôi muốn đặt mục tiêu tiết kiệm 50 triệu.'
  },
  {
    label: 'Tôi nhận lương tháng này',
    value: 'Tôi vừa nhận lương tháng này.'
  },
];

const ChatInterface = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I am CoverRun Financial Assistant. I can help you with your financial management, budgeting, and answering questions about the CoverRun product.',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [currentThinking, setCurrentThinking] = useState(null);
  const [thinkingHistory, setThinkingHistory] = useState([]);
  const [needNewThread, setNeedNewThread] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const { triggerRefresh } = useDataRefresh();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentThinking]);

  const handleSendMessage = async () => {
    if ((!inputMessage.trim() && !selectedImage) || isLoading) return;

    // Create user message content
    let messageContent = inputMessage;
    
    // If there's an image, show a placeholder in the UI
    if (selectedImage) {
      messageContent = inputMessage || 'Sent an image';
    }

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
      hasImage: !!selectedImage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setCurrentThinking(null);
    setThinkingHistory([]);

    try {
      // Get user email for identification
      const { data: { session } } = await supabase.auth.getSession();
      const userEmail = session?.user?.email;

      if (!userEmail) {
        throw new Error('User email not found. Please log in again.');
      }

      // Prepare request body
      const requestBody = {
        message: inputMessage || 'Analyze this image',
        conversation_history: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        user_email: userEmail,
        new_thread: needNewThread
      };

      // Add image data if available
      if (selectedImage) {
        const reader = new FileReader();
        const imagePromise = new Promise((resolve, reject) => {
          reader.onloadend = () => {
            // Get base64 data without the prefix (data:image/jpeg;base64,)
            const base64Data = reader.result.split(',')[1];
            const imageFormat = reader.result.split(',')[0].split(':')[1].split(';')[0];
            
            requestBody.image_data = base64Data;
            requestBody.image_format = imageFormat;
            resolve();
          };
          reader.onerror = reject;
          reader.readAsDataURL(selectedImage);
        });

        await imagePromise;
        // Clear the image after sending
        setSelectedImage(null);
        setImagePreview(null);
      }

      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      
      // Use streaming endpoint
      const response = await fetch(`${backendUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let finalResponse = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'thinking') {
                // Update current thinking step
                setCurrentThinking({
                  content: data.content,
                  step: data.step,
                  id: Date.now() + Math.random()
                });
                
                // Add to thinking history
                setThinkingHistory(prev => [...prev, {
                  content: data.content,
                  step: data.step,
                  id: Date.now() + Math.random()
                }]);
                
                // Clear current thinking after a delay (like GPT thinking mode)
                setTimeout(() => {
                  setCurrentThinking(null);
                }, 2000);
                
              } else if (data.type === 'final') {
                finalResponse = data.content;
                setCurrentThinking(null);
                
                const assistantMessage = {
                  id: Date.now() + 1,
                  role: 'assistant',
                  content: finalResponse,
                  timestamp: new Date()
                };
                setMessages(prev => [...prev, assistantMessage]);
                
              } else if (data.type === 'error') {
                throw new Error(data.content);
              } else if (data.type === 'done') {
                // Always trigger refresh after successful response in case tools were used
                console.log("ChatInterface: Streaming response completed, triggering refresh");
                triggerRefresh();
                // Reset the new thread flag after successful completion
                setNeedNewThread(false);
                break;
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'An error occurred, please try again later',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setCurrentThinking(null);
      // Reset the new thread flag in case of error
      setNeedNewThread(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/gif')) {
      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Add clear chat handler
  const handleClearChat = () => {
    setMessages([
      {
        id: 1,
        role: 'assistant',
        content: 'Hello! I am CoverRun Financial Assistant. I can help you with your financial management, budgeting, and answering questions about the CoverRun product.',
        timestamp: new Date()
      }
    ]);
    setCurrentThinking(null);
    setThinkingHistory([]);
    setNeedNewThread(true); // Flag to create new thread on next message
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-20 right-4 w-96 h-[500px] bg-white rounded-lg shadow-2xl border border-gray-200 flex flex-col z-50">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-green-600 text-white rounded-t-lg">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5" />
          <span className="font-semibold">CoverRun AI Assistant</span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={handleClearChat}
            variant="ghost"
            size="sm"
            className="text-white hover:bg-red-600 p-1 h-auto"
            title="Xóa hội thoại"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="text-white hover:bg-green-700 p-1 h-auto"
          >
            ✕
          </Button>
        </div>
      </div>

      {/* Suggestions */}
      {messages.length === 1 && messages[0].role === 'assistant' && (
        <div className="p-4 border-b border-gray-100 bg-gray-50">
          <div className="mb-2 text-sm text-gray-700 font-semibold">Gợi ý câu hỏi:</div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((s, idx) => (
              <button
                key={idx}
                className="bg-green-100 hover:bg-green-200 text-green-800 rounded-full px-3 py-1 text-xs font-medium transition"
                onClick={() => setInputMessage(s.value)}
                type="button"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="flex items-start gap-2">
                {message.role === 'assistant' && (
                  <Bot className="w-4 h-4 mt-0.5 flex-shrink-0" />
                )}
                <div className="text-sm leading-relaxed whitespace-pre-wrap">
                  {message.hasImage && (
                    <div className="flex items-center gap-1 mb-1 text-xs font-medium">
                      <Image className="w-3 h-3" /> 
                      <span>{message.role === 'user' ? 'Image sent' : 'Image analyzed'}</span>
                    </div>
                  )}
                  {message.content}
                </div>
                {message.role === 'user' && (
                  <User className="w-4 h-4 mt-0.5 flex-shrink-0" />
                )}
              </div>
              <div className={`text-xs mt-1 ${
                message.role === 'user' ? 'text-green-100' : 'text-gray-500'
              }`}>
                {message.timestamp.toLocaleTimeString('vi-VN', { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </div>
            </div>
          </div>
        ))}
        {/* Thinking display */}
        {currentThinking && (
          <div className="flex justify-start">
            <div className="bg-gray-50 rounded-lg p-3 max-w-[80%] border border-gray-200">
              <div className="flex items-center gap-2">
                <Bot className="w-4 h-4 text-gray-400" />
                <div className="text-sm text-gray-500 italic">
                  {currentThinking.content}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Loading indicator when no thinking is shown */}
        {isLoading && !currentThinking && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3 max-w-[80%]">
              <div className="flex items-center gap-2">
                <Bot className="w-4 h-4" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        {/* Image preview */}
        {imagePreview && (
          <div className="mb-2 relative">
            <div className="relative inline-block">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="h-20 rounded-md object-cover border border-gray-300" 
              />
              <button
                onClick={handleRemoveImage}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-md hover:bg-red-600 transition-colors"
                type="button"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
        
        <div className="flex gap-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here..."
            className="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
            rows="2"
            disabled={isLoading}
          />
          <div className="flex flex-col gap-2">
            {/* Image upload button */}
            <Button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-2 rounded-lg"
              title="Upload image"
            >
              <Image className="w-4 h-4" />
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/gif"
              onChange={handleImageSelect}
              className="hidden"
            />
            {/* Send button */}
            <Button
              onClick={handleSendMessage}
              disabled={(!(inputMessage.trim() || selectedImage)) || isLoading}
              className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 