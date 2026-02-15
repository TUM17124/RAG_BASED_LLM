import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();

  List<Map<String, String>> messages = [];
  bool isLoading = false;

  final String apiUrl = "http://127.0.0.1:8000/chat";


  Future<void> sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    setState(() {
      messages.add({"sender": "user", "text": text});
      isLoading = true;
    });

    _controller.clear();

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"question": text}),
      );

      final data = jsonDecode(response.body);

      setState(() {
        messages.add({"sender": "bot", "text": data["answer"]});
      });
    } catch (e) {
      setState(() {
        messages.add({"sender": "bot", "text": "Connection error"});
      });
    }

    setState(() => isLoading = false);
  }

  Widget buildMessage(Map<String, String> message) {
    bool isUser = message["sender"] == "user";

    return Container(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      padding: const EdgeInsets.all(8),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue : Colors.grey[300],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          message["text"] ?? "",
          style: TextStyle(
            color: isUser ? Colors.white : Colors.black,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Chatbot")),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: messages.length,
              itemBuilder: (context, index) {
                return buildMessage(messages[index]);
              },
            ),
          ),

          if (isLoading) const CircularProgressIndicator(),

          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration: const InputDecoration(
                    hintText: "Ask something...",
                  ),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.send),
                onPressed: () => sendMessage(_controller.text),
              )
            ],
          )
        ],
      ),
    );
  }
}
