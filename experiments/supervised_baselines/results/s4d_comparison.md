# Supervised S4D Comparison

The initial 12,000-step S4D run reached a best observed validation PER of
`0.37526`. A smaller 8,000-step sweep did not improve it: the baseline reached
`0.39058`, while lower learning rate, lower dropout, and wider variants were
worse. These values are validation-run minima, not held-out test estimates.

S4D is viable under the supervised recipe but was weaker than the tested S5
configuration. The available evidence is limited to a small, mostly
single-seed comparison.
