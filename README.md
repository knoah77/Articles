# Articles
Notes and code from articles
[link for how to create a good readme](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

# General Articles
<details>
  <summary>Steps to Become a Machine Learning Engineer</summary>
  
  
### [Steps to Become a Machine Learning Engineer](https://medium.com/cometheartbeat/7-steps-to-become-a-machine-learning-engineer-698cba0bc43c)

**what is machine learning**
- Sub field of Artificial inteligence that allows a machine to learn automatically and improve from experience
- techniquie for problem Solving and task automation

**Helpful libraries**
- Scikit-learn
- Tensorflow
- HuggingFace
- Comet

**3 key roles in data science projects**
- Data engineer
  - Create systems/pipelines to
    - Collect raw data
    -  manage data
    -  transform data into information
- Data scientist
  - Create Model Prototype
- ML engineer
  - Use tools to create models
  - Deploy them top production

**Machine Learning Project Lifecycle**
- Data Preperation
  - Clean Data using data preprocessing
    - Prevent garbage in garbage out
- Model Building
  - build the best model with the data they are provided from step above
    - Start with Simple models (regression)
    - Then move to complex models (neural networks)
  - Evaluate performance of the models (accuracy, precuision, recall, F1)
- Model Deployment
  - Deploy, Monitor and maintain the best model from above
    - put into production
  - Ensure that the model is making correct predictions 

**7 steps to becom an ML engineer**
1. Programming 
   - 2 primary languages 
     - R
     - Python
       - General Purpose
       - End to end machine learning projects 
       - cleaning to model deployment 
       - has the following frameworks
         - Pytorch
         - Skicit-Learn
         - Pyspark 
    
2. Machine learning Algorithms
   - Important to know algorithms to know when and what algorithms to use
   - 4 categories of algoisthms are
     - Regression (superviesd learning)
       - Linear Regression
       - Decision Trees
       - Support Vector Machines 
     - Classification (superviesd learning)
       - Logistic Regression
       - Naive Bayes
       - K-Nearest Neighbors 
     - Clustering (Unsuperviesd learning)
       - K-means
       - DBSCAN
       - Gaussian Mixture Models 
     - Dimensionallity Reduction (Unsuperviesd learning)
       - PCA
       - LDA
       - t-SNE 
3. Applied Mathmatics
   - Includes lots of applied mathmatics
     -  Statitics
     -  Linear Algebra
     -  Calculus
     -  Probability Theory
     -  Discrete Maths
   -  Applied when training the model coefficients
   -  Most are based in statistics
4. Deep Learning
   - ML models work well with small to medium datasets
   - Struggle with large datasets 
     - Deep Learning is used to handle these sets
       - Subset of ML that is an extension of artificial neural networks
       - examples of large datasets are
         - image classification
         - language to language techinques
           - GPT-3
           - BERT
       - Deep Learning is Black Box (dont know how they work)
       - Deep learning algorithms to know    
         - multilayer perception
         - convolutional neural networks
         - recurrent neural networks
         - long short-temr memory networks
         - generative adversarial networks
         - Transformers
         - Diffusion 
5. Machine learning Frameworks
   - Pandas
     - Good for data preprocessing
   - Matplotlib
     - Data Visulization
   - Seaborn
     - Data Visulization
   - Scikit-learn
     - implement machine learning algorithms
   - Tensorflow
     - deep learning analysis
   - Pytorch 
     - deep learning analysis
   - Comet
     - Model Optimization
6. MLOps (machine learning operations)
   - Putting machine learning into production
   - Bridge between model building and exporting the model to production
   - DEVOps equivalent for machine learning
   - Useful tools are
     - MLFLow
     - KubeFlow
     - MetaFlow
     - DataRobot
7. Cloud Computing
   - Cloud computing helps you to train models on powerful machines with multiple GPUs
   - deploy those models
   - run as many servers as you want
   - cloud computing services for machine learning are
     -  Amazon SageMaker
     -  Microsoft Azure Machine Learning
     -  GCP Vertex AI for ML engineering

**Additional Skills**
- Data Visualization
- SQL
- NoSQL
- PySpark
- Hadoop
- Docker
- Kubernetes
- CI-CD for Machine Learning
- Git and GitHub
- FastAPI
</details>

# Computer Vision
<details>
  <summary>Zero-shot image classification/segmentation/detection with CLIP</summary>

  
  ### [Zero-shot image classification/segmentation/detection with CLIP](https://medium.com/@khjfsdu/zero-shot-image-classification-segmentation-detection-with-clip-b8eec06582e3)

**OpenAI CLIP**
  - A model that processes images the same way as text
  - Treats image as a sequence of non-overlapping patches
    - Each patch is a visual token
    - Making an image a sequence of tokens
      - Once it is tokesn it can be processed using transformer
  - Trained on image caption pairs sourced from the web
  - How it Works
    - Converts image/text to vector embeddings using Contrastive loss
    - Generate image and text embeddings in the same vector space
      - allows computing of simularity of 
        - an image
        - a piece of text
      - does simulatitry comparision using cosine simularity between
        - image embedding and text embedding
  - converting to a vector embedding allows for AI by lowering data collectiopn and model training
  - Allows 0-shot prediction for
    - image classification
    - image segmentation
    - Object detection
  
**Image Classification**
  - Model is give
    - an image
    - Text (list of possible classes)
  - Model out puts a simularity to one of the possible classes
    - generates the image embedding
    - generates the texts embedding of the classes
      - picks the class with the embedding closest to the image embedding
   - [pseudo-code](https://github.com/openai/CLIP#zero-shot-prediction)
     ```
      # List of possible classes (text from above)
      classes = ["credit card", "driver's license", "passport"]

      # Loading the model
      model, preprocess = clip.load('ViT-B/32')

      # Preprocessing the data (image and then text)
      image_input = preprocess(image)
      text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes])

      # embedding the image and then the text
      image_features = model.encode_image(image_input)
      text_features = model.encode_text(text_inputs)

      # Pick the most similar class for the image
      similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
         ```
**Image Segmentation**
  - CLIPSeg Given
    - An image
    - Text
  - Can highlight in an image where that image representation of the text is in the given images
  - [Link](https://huggingface.co/blog/clipseg-zero-shot)
  
**Image Detection**
  - OWL-ViT does the above but returns a bounding box rather than outlines/shading
  - [Link](https://huggingface.co/spaces/adirik/OWL-ViT)
  
**Final Thoughs**
  - Speeds up time to create a model as training data is not needed
    - No data collection
    - No data labeling
    - No model training
  - Better for cases where you can tolerate a patentially higher error rate
  - Do need training for higher accuracy requirements
</details>


# Large Language models


# Synthetic Data
