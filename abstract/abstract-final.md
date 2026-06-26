%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KUZ CONFERRENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAMPLE LaTeX SOURCE (with no warranty)
\documentclass[twocolumn,letterpaper,10pt]{article}
\usepackage{meicogsci}
\usepackage{graphicx}				% include graphics
\usepackage[utf8]{inputenc}  				% for czech diacritics (used mainly in names)
%\usepackage[latin2]{inputenc}				% if utf8 is not working correctly
\usepackage[round,authoryear]{natbib}
%\usepackage[slovak]{babel}					% change english titles to czech
%\usepackage{slovak}							% in case of compiling with cslatex use this line instead the one above 
\usepackage[labelfont=bf]{caption}			% figure captions in bold
\usepackage[colorlinks=false]{hyperref} 	% hyperref support

\usepackage{lipsum}
\usepackage{titlesec}

% \makeatletter
% \def\convertto#1#2{\strip@pt\dimexpr #2*65536/\number\dimexpr 1#1}
% \makeatother

\setlength{\bibsep}{4pt plus 0.3ex}
\titlespacing*{\section} {0pt}{4ex}{2ex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% THE BEGINNING
\begin{document}

\date{}
\title{Instructions for Preparing Submissions for MEi:CogSci Conference}

\author{
\vspace{1ex}
\textbf{Author (Ethan Read)} \\ % another author from the same institution goes here
Institution\\
Institution Address\\
ethanread@student.elte.hu \\
% \and
% \vspace{1ex}
% \textbf{another author (from a different institution)}\\
% Different institution\\
% Different institution's address\\
% someone@other.institution.com \\
}

\maketitle 

\thispagestyle{empty}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INTRO
\section{Introduction}


The current surge of AI research has been enormous, clearing a path for rapid advancement in various fields.


Medicine is one such benefactor: recent years have seen the development of highly effective brain-to-speech implants for stroke patients who can no longer speak themselves (Willett et al, 2023). 



Up to 2025, the state of the art in speech decoders was restricted to recurrent neural networks (RNNs), a classic archtecture that has shown itself to be extremely well-generalizable. Despite being somewhat crude compared to newer architectures, RNNs are versatile and less data-hungry than more sophisticated models, keeping them a place in modern applications.



In 2026 Zhang et al published the Brain-to-Text Transformer (BIT), a first of its kind model that was trained on 400 hours of human and monkey Utah array data from a variety of tasks. Leveraging most available public primate Utah array data, they were able to train a transformer model that outperformed RNNs on attempted speech decoding, achieving a word error rate 


. To bridge the gap between different species and tasks, they trained the model in two stages: first in a self-supervised manner on unlabelled data to allow the model to learn general firing patterns, and then a second stage of supervised fine-tuning on data from stroke patients attempting speech. 


Although this result is impressive, there remain avenues of improvement. Transformers were designed for text-based translation tasks, turning one sequence into another. This works well for decoding signals in chunks, but other architectures might be better suited to real-time decoding of high-frequency recordings.
Transformers are very data-hungry and difficult to interpret. 

State-space models (SSMs) are a modern architecure intended for general sequence modelling, from text to audio to genomes. (Gu \& Dao, 2023). SSMs model the data as a dynamical system evolving in time, and thus have a structure that may more closely match the behaviour of nerual populations than that of an RNN or transformer. Because SSMs have a more restrictive represtantion of the data, they are more interpretable, require less data to train, and perform faster. 

If SSMs work well, they could offer both better computational efficiency and a clearer view of speech-related neural dynamics, including whether particular latent states track preparation, articulation, or pauses. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% METHODS
\section{Data \& Methods}


All data comes from publicly available Utah array recordings from human motor cortex, which are listed in Zhang et al. (2026). I will use a two-stage pipeline: self-supervised pretraining on pooled unlabeled recordings, followed by supervised fine-tuning on attempted-speech data from two paralyzed patients, one seen during pretraining and one held out. 


BIT was pre-trained by learning to reconstruct masked sections of the signal from surrounding context, so I will explore similar learning tasks with SSMs. I will also test contrastive training, a regime which forces the model to learn divergent representations of different signals. Since the findamental problem is phoneme classification, this task might help it identify patterns of different phonemes.



For the speech-decoding stage, I will attach a phoneme decoder and train with Connectionist Temporal Classification (CTC) loss, which allows variable-length phoneme sequences to be learned from neural time series without frame-level alignment. I will evaluate two SSM architectures: S5 and Mamba. If phonemes can be decoded from the signal, it is a relatively simple task to convert them into words.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RESULTS
\section{Hypotheses \& Discussion}

If the SSM encoder matches or exceeds the transformer baseline, it would suggest that speech-related Utah array activity is better modeled as a latent dynamical system than as a token-like sequence. If it does not, the result would still be informative because it would indicate that the available single-subject data are insufficient for the inductive bias that SSMs bring, despite their success on other long-sequence problems \citep{mamba2023}. In either case, the study would help determine whether these models are only accurate decoders or also useful tools for understanding the structure of speech-related neural activity. 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BIBLIOGRAPHY
% APA style
\bibliographystyle{apalike}
\bibliography{references}

%% the literature goes to references.bib in bibtex format
%% use pdflatex to compile

\end{document}
