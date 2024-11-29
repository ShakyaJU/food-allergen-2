import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';  // Use tflite_flutter instead of flutter_tflite
import 'package:image_picker/image_picker.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  File? filePath;
  String label = '';
  String allergen = '';
  String description = '';
  double confidence = 0.0;

  Map<String, dynamic>? allergenInfo;
  Map<String, dynamic>? classIndices;

  late Interpreter _interpreter;

  // Load the model and the resources
  Future<void> _tfLteInit() async {
    try {
      // Load the TensorFlow Lite model using tflite_flutter
      _interpreter = await Interpreter.fromAsset('assets/model_unquant.tflite');
      print('Model loaded successfully');

      // Load class indices
      String classIndicesData = await DefaultAssetBundle.of(context)
          .loadString("assets/class_indices.json");
      classIndices = json.decode(classIndicesData);
      print("Class Indices loaded successfully");

      // Load allergen map
      String allergenMapData =
      await DefaultAssetBundle.of(context).loadString("assets/class_allergen_map.json");
      allergenInfo = json.decode(allergenMapData);
      print("Allergen Info loaded successfully");
    } catch (e) {
      print("Error loading model or resources: $e");
    }
  }

  // Pick image from gallery
  pickImageGallery() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image == null) return;

    var imageMap = File(image.path);

    setState(() {
      filePath = imageMap;
    });

    // Run inference on the image
    var recognitions = await _predictImage(image.path);

    if (recognitions == null) {
      print("Recognition failed");
      return;
    }

    setState(() {
      confidence = (recognitions[0]['confidence'] * 100);
      label = recognitions[0]['label'].toString();
    });

    if (classIndices != null && allergenInfo != null) {
      // Process prediction info
      String predictedClass = classIndices![label] ?? 'Unknown';
      allergen = allergenInfo?[predictedClass]?['allergen'] ?? 'Unknown';
      description = allergenInfo?[predictedClass]?['description'] ?? 'No description available';
    }
  }

  // Pick image from camera
  pickImageCamera() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.camera);

    if (image == null) return;

    var imageMap = File(image.path);

    setState(() {
      filePath = imageMap;
    });

    // Run inference on the image
    var recognitions = await _predictImage(image.path);

    if (recognitions == null) {
      print("Recognition failed");
      return;
    }

    setState(() {
      confidence = (recognitions[0]['confidence'] * 100);
      label = recognitions[0]['label'].toString();
    });

    if (classIndices != null && allergenInfo != null) {
      // Process prediction info
      String predictedClass = classIndices![label] ?? 'Unknown';
      allergen = allergenInfo?[predictedClass]?['allergen'] ?? 'Unknown';
      description = allergenInfo?[predictedClass]?['description'] ?? 'No description available';
    }
  }

  // Run the model inference on an image
  Future<List<dynamic>?> _predictImage(String imagePath) async {
    try {
      var input = await _processImage(imagePath);

      var output = List.filled(1 * classIndices!.length, 0.0).reshape([1, classIndices!.length]);

      // Run inference
      _interpreter.run(input, output);

      return output[0];
    } catch (e) {
      print("Error during prediction: $e");
      return null;
    }
  }

  // Process image into a format suitable for inference
  Future<List<List<List<List<double>>>>> _processImage(String imagePath) async {
    // Your image processing logic here (resizing, normalization, etc.)
    // Example of loading and preprocessing the image
    // For now, returning a dummy input; Replace this with your actual preprocessing

    return [[[[]]]]; // This should be replaced with the actual input data
  }

  @override
  void dispose() {
    super.dispose();
    _interpreter.close();
  }

  @override
  void initState() {
    super.initState();
    _tfLteInit();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text("Food Allergen Detection"),
      ),
      body: Center(
        child: Column(
          children: [
            if (filePath != null)
              Padding(
                padding: const EdgeInsets.all(18.0),
                child: Card(
                  elevation: 20,
                  clipBehavior: Clip.hardEdge,
                  child: Column(
                    children: [
                      SizedBox(
                        height: 350,
                        width: double.infinity,
                        child: Image.file(
                          filePath!,
                          fit: BoxFit.cover,
                          alignment: Alignment.topCenter,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(12.0),
                        child: Text(
                          "Predicted Class: $label",
                          style: Theme.of(context).textTheme.titleLarge,
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Text(
                          "Confidence: ${confidence.toStringAsFixed(2)}%",
                          style: Theme.of(context).textTheme.titleSmall,
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Text(
                          "Allergen: $allergen",
                          style: Theme.of(context).textTheme.titleSmall,
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Text(
                          "Description: $description",
                          style: Theme.of(context).textTheme.titleSmall,
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              Padding(
                padding: const EdgeInsets.all(68.0),
                child: Image.asset(
                  "assets/catdog_transparent.png",
                ),
              ),
            ElevatedButton.icon(
                onPressed: pickImageGallery,
                label: const Padding(
                  padding: EdgeInsets.all(18.0),
                  child: Text("Pick an Image"),
                ),
                icon: const Icon(Icons.image)),
            const SizedBox(
              height: 12,
            ),
            ElevatedButton.icon(
                onPressed: pickImageCamera,
                label: const Padding(
                  padding: EdgeInsets.all(18.0),
                  child: Text("Snap a picture"),
                ),
                icon: const Icon(Icons.camera_alt)),
          ],
        ),
      ),
    );
  }
}
