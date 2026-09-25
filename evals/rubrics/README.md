# Style rubrics

One file per style, named `<style>.md`. The judge scores each `## Criterion`
from 1 to 5 and gives a one-line reason. Edit freely: criteria are parsed from
the `##` headings, and the text under each heading is shown to the judge as
the definition. Keep the anchors (what 1, 3 and 5 look like) concrete, since
they are what makes scores comparable across runs.

The judge never sees the style's name, the scene spec, code, file names, or
the generator's critique. It sees only the image and this rubric text. The
rubric describes a *tradition*, so it's fine that it names the medium.

Changing a rubric changes the scores. The report records a hash of each
rubric so runs with different rubrics are not silently compared.
