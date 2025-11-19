# Literature overview: WiFi RSSI localization

1. **Bahl & Padmanabhan, "RADAR: An In-Building RF-based User Location and Tracking System", IEEE INFOCOM 2000.**
   - First large-scale RSSI fingerprinting system. Introduces the idea of a fingerprint database and KNN/triangulation to locate a device indoors.
2. **Youssef & Agrawala, "The Horus WLAN Location Determination System", MobiSys 2005.**
   - Improves RADAR via probabilistic models per landmark, interpolation, and noise modeling to reduce sensitivity to RSSI fluctuations.
3. **X. Wang et al., "DeepFi: Deep Learning for Indoor Fingerprinting Using Channel State Information", IEEE TMC 2015.**
   - Uses deep neural networks to extract embeddings from CSI/RSSI before applying KNN, laying the foundations of the NN + KNN chain used here.
4. **C. Wu et al., "CSI-Net: A Unified Deep Neural Network for Indoor Localization", ACM BuildSys 2018.**
   - Shows that autoencoders/convolutional nets can produce compact embeddings that work well with localized search engines (KNN or clustering) for low-latency decisions.
5. **L. Sun et al., "Hybrid LSTM and kNN for WiFi Fingerprinting Localization", Sensors 2020.**
   - Describes a hybrid neural-network + KNN architecture operating in the latent space to improve resilience to noise, foreshadowing lightweight IoT-oriented L-KNN approaches.

These papers highlight three complementary trends:
- the need for a well-controlled grid and a dense fingerprint database,
- the benefit of neural networks for robust, discriminative representations,
- the value of local methods such as KNN or clustering to keep the decision explainable with low latency.
