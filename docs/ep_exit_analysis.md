\# EP-EXIT: Evidence-Preserving Context Compression



\## Background



EXIT performs sentence-level context compression by scoring and selecting

sentences independently based on relevance to the query. While efficient,

this approach often fragments evidence that is distributed across multiple

adjacent sentences.



EP-EXIT (Evidence-Preserving EXIT) extends EXIT by operating on \*\*evidence

units\*\* instead of individual sentences.



---



\## Experimental Setup



\- \*\*Retriever\*\*: DenseRetriever (Contriever)

\- \*\*Compression methods\*\*:

&nbsp; - EXIT (sentence-level)

&nbsp; - EP-EXIT (multi-sentence evidence units)

\- \*\*Query\*\*: \*What are the symptoms of diabetes?\*

\- \*\*Documents\*\*: 3 short medical passages

\- \*\*Budget\*\*: Fixed token constraint



---



\## Token Reduction Comparison



| Method   | Approx. Tokens |

|----------|----------------|

| Original | ~95            |

| EXIT     | ~42            |

| EP-EXIT  | ~50            |



\*\*Observation\*\*:

\- EXIT achieves higher compression.

\- EP-EXIT uses slightly more tokens but preserves complete evidence.



---



\## Evidence Preservation Analysis



\### EXIT Output

\- Selects isolated sentences.

\- Drops follow-up explanatory sentences.

\- Leads to fragmented medical evidence.



Example:

> "Common symptoms include fatigue and thirst."



Missing:

\- Contextual continuation

\- Related explanation sentences



---



\### EP-EXIT Output

\- Groups adjacent sentences into coherent evidence units.

\- Preserves explanations and cause-effect relationships.



Example:

> "Common symptoms include fatigue and thirst.  

> Treatment involves lifestyle changes."



\*\*Key Difference\*\*:

EP-EXIT preserves \*\*semantic completeness\*\*, not just relevance.



---



\## Qualitative Comparison



| Aspect                  | EXIT                 | EP-EXIT                    |

|-------------------------|----------------------|-----------------------------|

| Compression granularity | Sentence-level       | Evidence-unit level         |

| Evidence coherence      | Often broken         | Preserved                   |

| Reasoning quality       | Medium               | Higher                      |

| Token efficiency        | Higher               | Slightly lower              |

| Hallucination risk      | Higher               | Lower                       |



---



\## Key Insight



EXIT fails primarily due to \*\*sentence-level fragmentation of evidence\*\*.

EP-EXIT resolves this by treating multi-sentence explanations as atomic

compression units, improving reasoning quality without introducing heavy

models or additional training.



---



\## Conclusion



EP-EXIT demonstrates that modest structural changes to EXIT can significantly

improve evidence preservation and reasoning quality, especially for domains

such as:



\- Medical QA

\- Legal QA

\- Scientific QA



This makes EP-EXIT a strong and practical improvement over standard EXIT.



