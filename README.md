# Articles Notes
Notes and code from articles
- [link for how to create a good readme](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Nested Dropdowns](https://gist.github.com/ericclemmons/b146fe5da72ca1f706b2ef72a20ac39d#gistcomment-2694183)

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


<details>
  <summary>Google’s Sparrow Will Kill ChatGPT — It is Microsoft Teams vs. Slack All Over Again</summary>
[Google’s Sparrow Will Kill ChatGPT — It is Microsoft Teams vs. Slack All Over Again](https://medium.com/mlearning-ai/building-your-own-mini-chatgpt-b2e9716ab11](https://entreprenal.com/googles-sparrow-will-kill-chatgpt-it-is-microsoft-teams-vs-slack-all-over-again-da8c5a69c58f)
  

  
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


<details>
  <summary>Top 10 Object Detection Models of 2023</summary>
  ### [Top 10 Object Detection Models in 2023](https://medium.com/thelatestai/top-10-object-detection-models-in-2023-235acbc7d8b0)
  
 
  1. Yolov7
     - Pros
       - Very fast and efficient object detection
       - High accuracy on large datasets
       - Runs on low-end devices
     - Cons
       - Can struggle with small object detection
       - Requires a large dataset for optimal performance
  2. EfficientDet
     - Pros
       - State-of-the-art performance on several benchmark datasets
       - Efficient and accurate object detection
       - Can be trained on large datasets
     - Cons
       - Requires a large number of computational resources
       - Can be challenging to train on smaller datasets
  3. RetinaNet
     - Pros
       - Improved accuracy in object detection
       - Efficient and can run on low-end devices
       - Easy to train and use
     - Cons
       - Can struggle with small object detection
       - Requires a large amount of data for optimal performance
  4. Faster R-CNN
     - Pros
       - High accuracy in object detection
       - Effective for object detection in images and videos
       - Easy to train and use
     - Cons
       - Can be computationally expensive
       - Can be slow when detecting objects in real-time
  5. Mask R-CNN
     - Pros
       - High accuracy in object detection and instance segmentation
       - Can generate pixel-level masks for each detected object
       - Easy to train and use
     - Cons
       - Can be computationally expensive
       - Can be slow when detecting objects in real-time
  6. CenterNet
     - Pros
       - State-of-the-art performance on several benchmark datasets
       - High accuracy and efficiency in object detection
       - Can handle occluded and small objects
     - Cons
       - Can be computationally expensive
       - Can struggle with highly overlapping objects
  7. DETR
     - Pros
       - High accuracy and simplicity in object detection
       - Can handle highly overlapping objects
       - No anchor boxes or non-maximum suppression required
     - Cons
       - Can be computationally expensive
       - Requires a large amount of data for optimal performance
  8. Cascade R-CNN
     - Pros
       - State-of-the-art performance on several benchmark datasets
       - High accuracy in object detection
       - Can handle small and occluded objects
     - Cons
       - Can be computationally expensive
       - Requires a large amount of data for optimal performance
  9. SSD (Single Shot MultiBox Detector)
     - Pros
       - High accuracy and efficiency in object detection
       - Real-time object detection on low-end devices
       - Easy to train and use
     - Cons
       - Can struggle with small object detection
       - Can require a large dataset for optimal performance
  10. FCOS (Fully Convolutional One-Stage Object Detection)
     - Pros
       - State-of-the-art performance on several benchmark datasets
       - High accuracy and efficiency in object detection
       - No anchor boxes or non-maximum suppression required
     - Cons
       - Can be computationally expensive
       - Can require a large dataset for optimal performance
</details>
  
  
  
  










# Large Language models
<details>
  <summary>I tried making ChatGPT with GPT-3</summary>

### [I tried making ChatGPT with GPT-3](https://medium.com/geekculture/i-tried-making-chatgpt-with-gpt-3-4f0ef976d8c7)

- Difference between gpt-3 and chatgpt
  - ChatGPT remembers what prompt it received and what answer it gave
  - GPT-3 Does not, each generation is unrelated
- goal for the following code
  - Make GPT-3 aware of its past generations and prompts
  - Make the code fast
  - Make it so that it doesn’t exceed the accepted prompt length
  - sudo code for process
   ```
    
    # 1. input1 
    # 2. add input1 to model
    # 3. show output1
    # 4. input(2.....n)=take value of input1 take output1 
    # 5. add the input2 to the model 
    # 6. Show output(2...n)
    # 7. Loop to 4.
  ```
 Setting up the enviroment
 
```
pip install openai
```

actual code 
```
import openai 
  
# initial input
input_one=input("Enter your prompt: ")
  
# adding the inputs to the AI
c=input_one 
  
for i in range(0,1000):
  # running the GPT-3 API
  openai.api_key = "Your API Key"
  
  # Printing the input
  print(c)
  
  # model
  response = openai.Completion.create(
    engine="text-curie-001",
    prompt=c,
    temperature=0.7 ,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  
   # store and print the response
   b=response
   print("\n")
   print(response.choices[0].text)#output
  
  # the info generated and tell the AI later what it did a step before
  input_two="This was your last prompt: "+input_one+". This was the response you gave to the prompt:          "+response.choices[0].text+" remember this and answer the prompt given: "
    input_three=input("Enter your prompt: ")
    input_four=input_two+" "+input_three
    c=input_four
```
  
</details>

<details>
  <summary>Building Your Own Mini ChatGPT</summary>
[Building Your Own Mini ChatGPT](https://medium.com/mlearning-ai/building-your-own-mini-chatgpt-b2e9716ab119)
  

  
</details>

<details>
  <summary>Google’s Sparrow Will Kill ChatGPT — It is Microsoft Teams vs. Slack All Over Again</summary>
[Google’s Sparrow Will Kill ChatGPT — It is Microsoft Teams vs. Slack All Over Again](https://medium.com/mlearning-ai/building-your-own-mini-chatgpt-b2e9716ab11](https://entreprenal.com/googles-sparrow-will-kill-chatgpt-it-is-microsoft-teams-vs-slack-all-over-again-da8c5a69c58f)
  

  
</details>













# Synthetic Data












# Microcontrollers













# Side Hustles
<details>
  <summary>8 Stupidly Simple Programming Side Hustle That You Can Start in 2023 — No BS!</summary>
https://medium.com/geekculture/8-stupidly-simple-programming-side-hustle-that-you-can-start-in-2023-no-bs-93ec748d73ee

1. Technical Content writing
   - Pros
     - better understand the subject and identify any gaps in your knowledge
     - Make Connections
     - Can make money
   - How to get started
     - Where
       - LinkedIn, 
       - Twitter
       - Medium
       - Dev.to
       - Hashnode
      - Minimula amount of time
  
2. Building Online Products Using ChatGPT
   - examples
     - AI image Generator
     - Motivational Quote Generator
   - Benefits
     - Get job based off of project
     - More connections
  
3. Selling APIs
   - Create a way for different systems to communicate with each other and exchange data
   - Used in Finace, healthcare, e-commerence
   - How to Get started [Link](https://youtu.be/GK4Pl-GmPHk)
   - Can sell on RapidAI
   - Can sell Directly to clients 

4. Earning With Figma
   - UI/UX design
   - Free lancing 
     - Making designs for people directly
   - Selling design templates
     - Platforms such as Creative Market or Envato
   - Creating/Selling courses
     - platforms such as Udemy or Skillshare
   - Design/Sell Digital Products
     - items such as tshirts, stickers, phone cases 
     - Platforms such as Redbubble or society6

5. Earning with Canva
   - website design

6. Using Blockchain Technology
   - building a block chain

7. Become an Online Consultant
   - 

8. Selling a Programming Product
   - platforms such as Gumroad
</details>

