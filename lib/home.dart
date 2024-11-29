import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;

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

  // Load the model and resources
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
      confidence = (recognitions[0]['confidence'] as double) * 100;
      label = recognitions[0]['label'].toString();
    });

    if (classIndices != null && allergenInfo != null) {
      // Process prediction info
      String predictedClass = label;
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
      confidence = (recognitions[0]['confidence'] as double) * 100;
      label = recognitions[0]['label'].toString();
    });

    if (classIndices != null && allergenInfo != null) {
      // Process prediction info
      String predictedClass = label;
      allergen = allergenInfo?[predictedClass]?['allergen'] ?? 'Unknown';
      description = allergenInfo?[predictedClass]?['description'] ?? 'No description available';
    }
  }

  // Run the model inference on an image
  Future<List<dynamic>?> _predictImage(String imagePath) async {
    try {
      var input = await _processImage(imagePath);

      var output = List.filled(classIndices!.length, 0.0).reshape([1, classIndices!.length]);

      // Run inference
      _interpreter.run(input, output);

      // Process output: Get predictions
      int predictedIndex = output[0].indexOf(output[0].reduce((a, b) => a > b ? a : b));
      double confidence = output[0][predictedIndex];
      String predictedLabel = classIndices![predictedIndex.toString()] ?? 'Unknown';

      return [
        {"label": predictedLabel, "confidence": confidence},
      ];
    } catch (e) {
      print("Error during prediction: $e");
      return null;
    }
  }
  Future<List<List<List<List<double>>>>> _processImage(String imagePath) async {
    try {
      // Load the image as bytes
      final imageBytes = File(imagePath).readAsBytesSync();

      // Decode the image
      img.Image? originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) {
        throw Exception("Failed to decode image");
      }

      // Resize the image to the input size for the model (e.g., 224x224)
      img.Image resizedImage = img.copyResize(originalImage, width: 224, height: 224);

      // Convert image to normalized tensor
      List<List<List<List<double>>>> input = List.generate(
        1,
            (_) => List.generate(
          224,
              (y) => List.generate(
            224,
                (x) {
              // Get pixel color
              final pixel = resizedImage.getPixel(x, y);

              // Extract RGB components from the pixel
              double r = pixel.r / 255.0; // Red component
              double g = pixel.g / 255.0; // Green component
              double b = pixel.b / 255.0; // Blue component

              // Return the normalized RGB values
              return [r, g, b];
            },
          ),
        ),
      );

      return input;
    } catch (e) {
      print("Error processing image: $e");
      return [];
    }
  }

  @override
  void dispose() {
    try {
      _interpreter.close();
    } catch (e) {
      print("Error closing interpreter: $e");
    }
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _tfLteInit();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Food Allergen Classifier')),
      body: Center(
        child: Column(
          children: [
            if (filePath != null)
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
            ElevatedButton.icon(
              onPressed: pickImageGallery,
              icon: const Icon(Icons.photo),
              label: const Text("Pick Image from Gallery"),
            ),
            ElevatedButton.icon(
              onPressed: pickImageCamera,
              icon: const Icon(Icons.camera),
              label: const Text("Pick Image from Camera"),
            ),
          ],
        ),
      ),
    );
  }
}
