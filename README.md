# decisionmaking-code-py

[![Python package](https://github.com/griffinbholt/decisionmaking-code-py/actions/workflows/python-package.yml/badge.svg)](https://github.com/griffinbholt/decisionmaking-code-py/actions/workflows/python-package.yml)

*[Original Julia Code](https://github.com/algorithmsbooks/decisionmaking-code) by: Mykel Kochenderfer, Tim Wheeler, and Kyle Wray*

*Python Versions by: Griffin Holt*

Python versions of all typeset code blocks from the book, [Algorithms for Decision Making](https://algorithmsbook.com/).

I share this content in the hopes that it helps you and makes the decision making algorithms more approachable and accessible (especially to those not as familiar with Julia). Thank you for reading!

If you encounter any issues or have pressing comments, please [file an issue](https://github.com/griffinbholt/decisionmaking-code-py/issues/new/choose). (There are likely to still be bugs as I have not finished testing all of the classes and functions.)

**Note:** Rewriting all of the typeset code blocks from the book in Python has convinced me of one thing: Julia was the correct choice of programming language for code in the book. Many aspects of Julia's structure (including typing, inclusion of unicode characters, anonymous functions, mutable function declaration, etc.) make it much more ideal for communicating an algorithm than Python. If you are a student in Mykel's AA228/CS238 course or a reader that wishes to dive deep into sequential decision making, I would recommend you just learn Julia's syntax. However, if you need a quick Pythonic reference, that's what this library is for. (This has also been a great exercise for me to ensure and deepen my understanding of the material.)

## Progress Update: (26 Oct 2023)

| Chapter(s) | Written | Tested | Notes |
|--:|:--|:--|:--|
|  1 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  2 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  3 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  4 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  5 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  6 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  7 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  8 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  9 | ▌▌▌▌▌▌▌▌▌ 99% | 0% | Just need to make a few syntactic decisions |
| 10 | ▌▌▌▌▌▌▌ 67% | 0% | `EvolutionStrategies` and `IsotropicEvolutionStrategies` need to be written |
| 11 | ▌▌▌▌▌▌▌▌▌ 99% | 0% | Just need to make a few syntactic decisions |
| 12 | ▌▌▌▌▌ 50% | 0% | `TrustRegionUpdate` and `ClampedSurrogateUpdate` need to be written |
| 13 | 0% | 0% | Skipped for now; coming back to it later |
| 14 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
| 15 | ▌▌▌▌▌▌▌▌▌ 90% | 0% | Just need to write the `simulate` method from the end of the chapter |
| 16 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
| 17 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
| 18 | ▌▌▌▌▌▌▌ 70% | 0% | `MaximumEntropyIRL` needs to be written |
| 19 | ▌▌▌▌ 40% | 0% | `UnscentedKalmanFilter` and the particle filters need to be written |
| 20 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
| 21 | ▌▌▌▌▌▌▌▌ 80% | 0% | `SawtoothHeuristicSearch`, `TriangulatedPolicy`, and `TriangulatedIteration` need to be written |
| 22 | 0% | 0% | Nothing written |
| 23 | 0% | 0% | Nothing written |
| 24 | 0% | 0% | Nothing written |
| 25 | 0% | 0% | Nothing written |
| 26 | 0% | 0% | Nothing written |
| 27 | 0% | 0% | Nothing written |

I have also written code for pertinent examples and exercises through Chapter 7.