# Vidhi.ai - Chat with Legal PDF

Welcome to **Vidhi.ai**, a legal assistance agent that also includes a feature to upload and interact with your legal PDFs! This feature allows you to upload shorter legal documents (PDFs) and ask questions related to the content of the document. The agent will provide relevant answers based on the uploaded PDF. This project is deployed using **Gradio** on **Hugging Face Spaces**.

## Features

- **Legal PDF Interaction**: Upload your legal PDFs and ask questions related to the document.
- **Legal Assistance**: Get answers to your legal queries based on the content of the uploaded PDF.
- **User-Friendly Interface**: Easy-to-use interface for seamless interaction.
- **Deployed on Hugging Face Spaces**: Accessible via a web interface.

## Getting Started

To clone and run this project locally, follow the steps below.

### Prerequisites

- Ensure you have **Git LFS** installed. You can download it from [here](https://git-lfs.com).

### Cloning the Repository

You can clone the repository using either HTTPS or SSH.

#### HTTPS

1. Install Git LFS:
   ```bash
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/ayushraj0906/ChatWith_LegalPDF
   ```

3. If you want to clone without large files (just their pointers):
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/ayushraj0906/ChatWith_LegalPDF
   ```

#### SSH

1. Install Git LFS:
   ```bash
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone git@hf.co:spaces/ayushraj0906/ChatWith_LegalPDF
   ```

3. If you want to clone without large files (just their pointers):
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:spaces/ayushraj0906/ChatWith_LegalPDF
   ```

## Usage

Once you have cloned the repository, you can run the Gradio app locally to interact with the chatbot.

1. Navigate to the project directory:
   ```bash
   cd ChatWith_LegalPDF
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Gradio app:
   ```bash
   python app.py
   ```

4. Open your web browser and go to `http://127.0.0.1:7860` to interact with the agent.

## How to Use the Chat with Legal PDF Feature

1. **Upload a PDF**: Use the file uploader to upload your legal PDF. Please ensure the PDF is as short as possible for optimal performance.
2. **Ask Questions**: Once the PDF is uploaded, you can start asking questions related to the content of the PDF.
3. **Get Answers**: The agent will provide answers based on the content of the uploaded PDF.

## Contributing

We welcome contributions to improve Vidhi.ai! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license. Please refer to [LICENSE] file.

## Acknowledgments

- **Gradio** for providing an easy-to-use interface for deploying machine learning models.
- **Hugging Face Spaces** for hosting the chatbot.
- The open-source community for their continuous support and contributions.

## Contact

For any questions or feedback, feel free to reach out to [Ayush Raj](mailto:ayush.raj.bme22@iitbhu.ac.in).

---

Thank you for using Vidhi.ai! We hope this tool helps you navigate the legal landscape in India with ease and confidence.
