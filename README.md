## 🧠 BlockCode — Visual Neural Network Builder  

BlockCode is a **PyGame-based interactive platform** that lets users design and train neural networks visually, **without writing any code**.  
Each block represents a functional component of a machine learning workflow, such as a linear layer, activation function, or optimizer. By connecting these blocks, users can build an entire neural network pipeline and watch how data and gradients move through it in real time.  

The project was created to make **machine learning workflows accessible and intuitive**. It bridges deep learning and visual programming, showing how concepts like forward propagation, backpropagation, and gradient descent actually unfold inside a model.  

---

### 🎯 Objectives  
- Enable anyone to create and experiment with neural networks through an interactive, **no-code interface**.  
- Visualize how activations, gradients, and weights evolve during training.  
- Rebuild the logic of frameworks like PyTorch from scratch to deepen understanding of neural network mechanics.  

---

### ⚙️ Key Features  

#### 🧩 Visual Workflow Creation  
- Drag-and-drop block interface for building complete neural network architectures.  
- Each block represents a layer, activation, loss, or optimizer, forming a directed computation graph when connected.  
- Designed for interactive education and creative experimentation.  

#### 🔁 Custom Backpropagation Engine  
- Fully implemented **forward and backward propagation** with a mix of NumPy and PyTorch.  
- Real-time visualization of gradient flow, weight updates, and error propagation.  
- Adjustable learning rate, step size, and update frequency for hands-on exploration.  

#### 🔬 Real-Time Feedback  
- Displays activations, gradients, and weights directly within each block.  
- Color-coded connection lines indicate data flow and gradient direction.  
- Offers a live understanding of how neural networks train and converge.  

#### 🧱 Modular and Extensible Design  
- Built around a modular `Block` class that defines computation, connections, and rendering.  
- New block types can be easily added to extend functionality.  
- Architecture supports convolutional, normalization, and custom operation blocks in future updates.  

---

### 🧠 Tech Stack  
Python • PyGame • NumPy • Neural Networks • Gradient Descent  

---

### 🚀 Vision  
BlockCode aims to become a **visual playground for neural networks**, lowering the barrier to entry for new learners and enabling researchers to explore complex ideas through an interactive, intuitive interface.
