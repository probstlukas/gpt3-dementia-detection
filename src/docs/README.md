<!-- Template from https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/teco-kit">
    <img src="images/teco.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Prediction of dementia based on speech using LLMs</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/probstlukas/TECO"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/probstlukas/TECO">View Demo</a>
    ·
    <a href="https://github.com/probstlukas/TECO/issues">Report Bug</a>
    ·
    <a href="https://github.com/probstlukas/TECO/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was created during my position as a research assistant at TECO research group at Karlsruhe Institute of Technology (KIT).
The objective was to detect dementia based on speech by harnessing the power of AI methods.
So far this project provides two kinds of inputs to feed a machine learning model to be trained on: [GPT-3 Text Embeddings](https://platform.openai.com/docs/guides/embeddings) and acoustic features with [openSMILE](https://github.com/audeering/opensmile). 

### GPT-3 Text Embeddings
TODO

It's not necessary to scale the embeddings before using them. They are already normalised and are in the vector space with a certain distribution.

### Acoustic Features
TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

In order to get access to the data used in this project, you must first join as a [DementiaBank member](https://dementia.talkbank.org/index.html).

For the sake of simplicity, I've used the data set supplied for the [ADReSSo-challenge](https://dementia.talkbank.org/ADReSS-2021/) which has been balanced with respect to age and gender in order to eliminate potential confunding and bias.

In my case, the downloaded ADReSSo audio files had an incompatible format to transcribe them with Whisper. Therefore I had to format them with ffmepg first. Since we cannot reformat the files and replace them at the same time, we have to save them temporarily and replace the old files afterwards:
```
find . -name '*.wav' -exec sh -c 'mkdir -p fix && ffmpeg -i "$0" "fix/$(basename "$0")"' {} \;
```

### Installation

1. Get an OpenAI API Key at [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
2. Set an environment variable 'OPENAI\_API\_KEY' (replace ~/.zshrc with ~/.bashrc if you use Bash):
    ```sh
    echo "export OPENAI\_API\_KEY='your key'" | cat >> ~/.zshrc
    ```
1. Clone the repo
   ```sh
   git clone https://github.com/probstlukas/TECO.git
   ```
2. Install required Python packages
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/probstlukas/TECO/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [DementiaBank](https://dementia.talkbank.org/)
* [ADReSSo-Challenge](https://dementia.talkbank.org/ADReSS-2021/)
* ["Predicting dementia from spontaneous speech using large language models" by Felix Agbavor and Hualou Liang](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000168)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/probstlukas/TECO.svg?style=for-the-badge
[contributors-url]: https://github.com/probstlukas/TECO/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/probstlukas/TECO.svg?style=for-the-badge
[forks-url]: https://github.com/probstlukas/TECO/network/members
[stars-shield]: https://img.shields.io/github/stars/probstlukas/TECO.svg?style=for-the-badge
[stars-url]: https://github.com/probstlukas/TECO/stargazers
[issues-shield]: https://img.shields.io/github/issues/probstlukas/TECO.svg?style=for-the-badge
[issues-url]: https://github.com/probstlukas/TECO/issues
[license-shield]: https://img.shields.io/github/license/probstlukas/TECO.svg?style=for-the-badge
[license-url]: https://github.com/probstlukas/TECO/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/thelukasprobst
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/Python-2023
[Python-url]: https://python.org 
