import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:lottie/lottie.dart';
import 'package:record/record.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:http_parser/http_parser.dart';  // for MediaType

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  List<Map<String, dynamic>> messages = [];
  String? currentSessionId;
  bool isChatLoading = false;
  bool isSoundLoading = false;
  bool isBreedLoading = false;
  bool isRecording = false;

  Timer? _recordingTimer;
  int recordingSeconds = 0;
  final AudioRecorder _recorder = AudioRecorder();

  final String chatUrl = "http://10.0.2.2:8000/chat";
  final String soundUrl = "http://10.0.2.2:8000/predict-cat-sound";
  final String breedUrl = "http://10.0.2.2:8000/predict-cat-breed";

  static const int maxRecordingSeconds = 12;
  static const int maxFileSizeBytes = 8 * 1024 * 1024; // 8 MB (WAV is larger)

  @override
  void initState() {
    super.initState();
    _loadSessionId();
  }

  Future<void> _loadSessionId() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      currentSessionId = prefs.getString('cat_chat_session_id');
    });
  }

  Future<void> _saveSessionId(String? id) async {
    if (id == null) return;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('cat_chat_session_id', id);
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    _recordingTimer?.cancel();
    _recorder.stop(); // safe even if not recording
    super.dispose();
  }

  Future<void> sendMessage() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      messages.add({"sender": "user", "text": text});
      isChatLoading = true;
    });
    _textController.clear();
    _scrollToBottom();

    try {
      final response = await http
          .post(
            Uri.parse(chatUrl),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({
              "question": text,
              "session_id": currentSessionId,
            }),
          )
          .timeout(const Duration(seconds: 200));

      if (response.statusCode != 200) {
        throw Exception("Server returned ${response.statusCode}");
      }

      final data = jsonDecode(response.body);
      final answer = (data["answer"] as String?)?.trim() ?? "No response received.";
      final newSid = data["session_id"] as String?;

      if (newSid != null && newSid != currentSessionId) {
        currentSessionId = newSid;
        await _saveSessionId(newSid);
      }

      setState(() {
        messages.add({"sender": "bot", "text": answer});
        isChatLoading = false;
      });
      _scrollToBottom();
    } catch (e) {
      _addBotMessage("Connection failed: $e");
      setState(() => isChatLoading = false);
    }
  }

  Future<void> _startRecording() async {
    if (isRecording || isSoundLoading) return;

    var micStatus = await Permission.microphone.request();
    if (!micStatus.isGranted) {
      if (micStatus.isPermanentlyDenied) {
        await openAppSettings();
      }
      _addBotMessage("Microphone permission required.");
      return;
    }

    setState(() {
      messages.add({"sender": "user", "text": "🎤 Recording cat sound..."});
      isRecording = true;
      recordingSeconds = 0;
    });
    _scrollToBottom();

    final tempDir = await getTemporaryDirectory();
    final path = '${tempDir.path}/cat_sound_${DateTime.now().millisecondsSinceEpoch}.wav';

    try {
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,   // WAV PCM 16-bit — best compatibility
          sampleRate: 44100,
          numChannels: 1,
        ),
        path: path,
      );

      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (!mounted) return;
        setState(() => recordingSeconds++);
        if (recordingSeconds >= maxRecordingSeconds) {
          _stopAndPredict();
        }
      });
    } catch (e) {
      _addBotMessage("Failed to start recording: $e");
      setState(() => isRecording = false);
    }
  }

  Future<void> _stopAndPredict() async {
  if (!isRecording) return;

  _recordingTimer?.cancel();
  setState(() => isRecording = false);

  String? rawPath;
  try {
    rawPath = await _recorder.stop();
    if (rawPath == null || rawPath.isEmpty) {
      throw Exception("No audio file created");
    }

    final rawFile = File(rawPath);
    if (!await rawFile.exists()) {
      throw Exception("Recorded file missing");
    }

    final rawBytes = await rawFile.readAsBytes();
    final fileSize = rawBytes.length;

    print('Raw recorded: $rawPath | Size: $fileSize bytes');

    if (fileSize > maxFileSizeBytes) {
      await rawFile.delete();
      throw Exception("File too large");
    }
    if (fileSize < 16000) {
      await rawFile.delete();
      throw Exception("Recording too short");
    }

    // Create proper WAV with header
    final sampleRate = 44100;
    final channels = 1;
    final bitsPerSample = 16;
    final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
    final blockAlign = channels * bitsPerSample ~/ 8;
    final dataSize = fileSize;
    final fileSizeTotal = 36 + dataSize;

    final header = Uint8List(44)
      ..setRange(0, 4, [82, 73, 70, 70]) // "RIFF"
      ..buffer.asByteData().setUint32(4, fileSizeTotal, Endian.little)
      ..setRange(8, 12, [87, 65, 86, 69]) // "WAVE"
      ..setRange(12, 16, [102, 109, 116, 32]) // "fmt "
      ..buffer.asByteData().setUint32(16, 16, Endian.little) // Subchunk1Size
      ..buffer.asByteData().setUint16(20, 1, Endian.little) // AudioFormat (PCM = 1)
      ..buffer.asByteData().setUint16(22, channels, Endian.little)
      ..buffer.asByteData().setUint32(24, sampleRate, Endian.little)
      ..buffer.asByteData().setUint32(28, byteRate, Endian.little)
      ..buffer.asByteData().setUint16(32, blockAlign, Endian.little)
      ..buffer.asByteData().setUint16(34, bitsPerSample, Endian.little)
      ..setRange(36, 40, [100, 97, 116, 97]) // "data"
      ..buffer.asByteData().setUint32(40, dataSize, Endian.little);

    final wavBytes = Uint8List(header.length + rawBytes.length)
      ..setRange(0, header.length, header)
      ..setRange(header.length, header.length + rawBytes.length, rawBytes);

    // Save as new valid WAV file
    final tempDir = await getTemporaryDirectory();
    final wavPath = '${tempDir.path}/valid_cat_sound_${DateTime.now().millisecondsSinceEpoch}.wav';
    final wavFile = File(wavPath);
    await wavFile.writeAsBytes(wavBytes);

    print('Created valid WAV: $wavPath | Size: ${await wavFile.length()} bytes');

    // Upload the fixed file
    setState(() {
      isSoundLoading = true;
      messages.add({"sender": "bot", "text": "🔊 Analyzing cat sound..."});
    });
    _scrollToBottom();

    var request = http.MultipartRequest('POST', Uri.parse(soundUrl));
    final multipartFile = await http.MultipartFile.fromPath(
      'file',
      wavPath,
      contentType: MediaType('audio', 'wav'),
    );
    request.files.add(multipartFile);

    final streamedResp = await request.send().timeout(const Duration(seconds: 20000));
    final statusCode = streamedResp.statusCode;

    final respStr = await streamedResp.stream.bytesToString();

    if (statusCode != 200) {
      throw Exception("Server error $statusCode: $respStr");
    }

    final data = jsonDecode(respStr);

    setState(() {
      messages.add({
        "sender": "bot",
        "text":
            "**Detected:** ${data['sound'] ?? 'Unknown'}\n"
            "**Confidence:** ${data['confidence'] ?? 'N/A'}\n\n"
            "${data['explanation'] ?? 'No explanation available.'}",
      });
      isSoundLoading = false;
    });
    _scrollToBottom();

    // Cleanup
    await rawFile.delete();
    await wavFile.delete();
  } catch (e) {
    _addBotMessage("Sound analysis failed: $e");
    setState(() => isSoundLoading = false);
    // Cleanup any leftover files
    if (rawPath != null && await File(rawPath).exists()) await File(rawPath).delete();
  }
}

  Future<void> _pickAndPredictBreed() async {
    // unchanged — assuming breed endpoint expects 'image' field
    if (isBreedLoading) return;

    var status = await Permission.photos.request();
    if (!status.isGranted) {
      if (await Permission.photos.isPermanentlyDenied) await openAppSettings();
      _addBotMessage("Gallery permission required.");
      return;
    }

    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() {
      messages.add({"sender": "user", "text": "📸 Analyzing photo..."});
      isBreedLoading = true;
    });
    _scrollToBottom();

    try {
      var request = http.MultipartRequest('POST', Uri.parse(breedUrl));
      request.files.add(await http.MultipartFile.fromPath('image', image.path));

      final streamed = await request.send();
      if (streamed.statusCode != 200) {
        final err = await streamed.stream.bytesToString();
        throw Exception("Server error ${streamed.statusCode}: $err");
      }

      final respStr = await streamed.stream.bytesToString();
      final data = jsonDecode(respStr);

      setState(() {
        messages.add({
          "sender": "bot",
          "text":
              "**Predicted breed:** ${data['breed'] ?? 'Unknown'}\n"
              "**Confidence:** ${data['confidence'] ?? 'N/A'}",
        });
        isBreedLoading = false;
      });
      _scrollToBottom();
    } catch (e) {
      _addBotMessage("Breed prediction failed: $e");
      setState(() => isBreedLoading = false);
    }
  }

  void _addBotMessage(String text) {
    setState(() {
      messages.add({"sender": "bot", "text": text});
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 400),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Widget _buildMessage(Map<String, dynamic> msg) {
    final isUser = msg["sender"] == "user";
    final text = msg["text"] as String? ?? "";

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue[700] : Colors.grey[100],
          borderRadius: BorderRadius.circular(20),
          boxShadow: isUser
              ? null
              : [BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 8)],
        ),
        child: isUser
            ? Text(text, style: const TextStyle(color: Colors.white, fontSize: 16))
            : MarkdownBody(
                data: text,
                styleSheet: MarkdownStyleSheet(
                  p: const TextStyle(fontSize: 16, color: Colors.black87),
                  strong: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final anyLoading = isChatLoading || isSoundLoading || isBreedLoading || isRecording;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Cat Care Assistant 🐾"),
        elevation: 0,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.symmetric(vertical: 12),
              itemCount: messages.length + (anyLoading ? 1 : 0),
              itemBuilder: (context, index) {
                if (index < messages.length) {
                  return _buildMessage(messages[index]);
                }
                return Padding(
                  padding: const EdgeInsets.all(32),
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Lottie.asset(
                          'assets/animations/cat_scratch.json',
                          width: 140,
                          height: 140,
                          repeat: true,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          isRecording
                              ? "Recording... $recordingSeconds / $maxRecordingSeconds s"
                              : "Analyzing...",
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.grey,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        if (isRecording) ...[
                          const SizedBox(height: 20),
                          ElevatedButton.icon(
                            onPressed: _stopAndPredict,
                            icon: const Icon(Icons.stop, color: Colors.white),
                            label: const Text("Stop & Analyze"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.red[700],
                              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 20),
            child: Row(
              children: [
                IconButton(
                  icon: Icon(
                    isRecording ? Icons.stop_circle_rounded : Icons.mic_rounded,
                    color: isRecording ? Colors.red : Colors.blue,
                    size: 32,
                  ),
                  onPressed: anyLoading && !isRecording ? null : _startRecording,
                ),
                IconButton(
                  icon: const Icon(Icons.photo_camera_rounded, color: Colors.blue, size: 28),
                  onPressed: anyLoading ? null : _pickAndPredictBreed,
                ),
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: InputDecoration(
                      hintText: "Ask about your cat...",
                      filled: true,
                      fillColor: Colors.grey[200],
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => sendMessage(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  icon: const Icon(Icons.send),
                  onPressed: anyLoading ? null : sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}