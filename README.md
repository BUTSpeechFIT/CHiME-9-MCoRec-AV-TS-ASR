# BUT CHiME-9 MCoRec Submission Implementation

This repository contains BUT Speech [CHiME-9 MCoRec](https://www.chimechallenge.org/challenges/chime9/task1/index) submission.

Our system overall achieved **3rd place**, only 0.16\% WER and 0.5\% F1 behind the best-performing system.

## Setup
After cloning the repository, init the submodules by running:
```bash
git submodule update --init --recursive
```

## Run
1. Visit `asr_model/README.md` to setup the ASR model. 
2. Visit `conversation_clustering/README.md` to setup the conversation clustering pipeline.
3. Run the inference on MCoRec `dev` set to obtain `.vtt` transcription in the format of the challenge.
4. Follow the `conversation_clustering/README.md` to run the clustering using the inferred transcripts.

## 📚 Citation
If you use our models or code, please cite the following works:
```
@misc{klement2026descriptionchime9mcorecchallenge,
      title={BUT System Description for CHiME-9 MCoRec Challenge}, 
      author={Dominik Klement and Alexander Polok and Nguyen Hai Phong and Prachi Singh and Lukáš Burget},
      year={2026},
      eprint={2604.27436},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2604.27436}, 
}
```

## 🤝 Contributing

Contributions are welcome.
If you’d like to improve the code, add new features, or extend the training pipeline, please open an issue or submit a pull request.

---

## 📬 Contact

For questions or collaboration, please contact: 
- [iklement@fit.vut.cz](mailto:iklement@fit.vut.cz) (ASR)
- [xnguye28@stud.fit.vut.cz](mailto:xnguye28@stud.fit.vut.cz) (Conversation Clustering)