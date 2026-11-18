# Recipe Entity Recognition (CRF)

**Question:** can free-text recipe ingredient lists be turned into structured data (quantity, unit, ingredient), for example to build a shopping list automatically?

I trained a Conditional Random Field (CRF) on 280 annotated recipes to tag every word as an **ingredient**, **quantity** or **unit**, then dug into where it goes wrong.

## Results (84 unseen validation recipes, 2,876 tokens)

| Label | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Ingredient | 0.98 | 1.00 | 0.99 |
| Quantity | 1.00 | 0.98 | 0.99 |
| Unit | 0.97 | 0.90 | 0.93 |
| **Overall** | | | **macro F1 0.97 · accuracy 98.2%** |

**The 98% hides a weak spot.** Units caused 37 of the 53 errors, mostly by being tagged as ingredients. Error analysis traced them to:

- ambiguous words ("cloves" can be a unit or part of an ingredient, and "gram" appears in "black gram")
- inconsistent labels in the source data (preparation words like "diced" and "finely" sometimes labelled as units)
- quantity formats the regex doesn't cover ("6-8", "a few")

The full write-up is in [`report/recipe_entity_recognition_report.pdf`](report/recipe_entity_recognition_report.pdf).

## Approach

1. **Validation:** checked that tokens and labels line up and removed 5 misaligned recipes (285 → 280).
2. **Split:** 70/30 at recipe level (196 train, 84 validation) so recipes stay intact.
3. **Features:** for each token, spaCy lemma, part of speech and dependency; word shape; a quantity regex for numbers, fractions and decimals; unit and quantity keyword lists learned from training data; a cooking-method flag; and the neighbouring words.
4. **Class imbalance:** ingredients are about 75% of tokens. Because `sklearn-crfsuite` has no class-weight option, I used weighted resampling of whole training recipes. This is an approximation, and I didn't run a controlled comparison to prove it helped.
5. **Model:** CRF (L-BFGS, c1=0.5, c2=1.0, 100 iterations).

## Data

The course-provided annotated recipe file (`ingredient_and_quantity.json`) is not included. See [`data/README.md`](data/README.md).

## Tech

Python · sklearn-crfsuite · spaCy · scikit-learn · pandas · Matplotlib · Seaborn

---

Completed as part of the upGrad Executive Diploma in Data Science & AI. Built by **Tara Joseph** · [LinkedIn](https://www.linkedin.com/in/joseph-tara)
