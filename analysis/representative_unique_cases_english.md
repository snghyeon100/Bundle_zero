# Representative Unique-View Cases

These cases are selected because the ground-truth item does not appear as an exact copied item in the successful view's context, but the successful view still provides a clear view-specific signal.

## Case 1: IB x BI Unique Success

Dataset: POG-Dense  
Bundle ID: 27788  
Successful view: IB x BI  
Ranks: IB x BI = 1, IU x UI = 8, BI x IB = 3

### Input

1. Vintage Hong Kong-style black chain crossbody bag
2. French-style wool beret

### Ground Truth

H. Korean-style chunky heeled ankle boots / Martin boots

### Candidates

A. Men's red graphic print T-shirt  
B. Short knit cardigan  
C. Korean-style leather bucket bag / large shoulder bag  
D. French-style small square leather handbag  
E. Short printed tie-front dress  
F. Mini chain crossbody bag  
G. Statement fashion earrings  
H. Korean-style chunky heeled ankle boots / Martin boots  
I. Color-block leather bucket/tote bag  
J. Short red double-faced wool/alpaca coat

### IB x BI Additional Context

Input item 1, the black crossbody bag, is frequently included in outfits with:

- Winter fleece-lined heeled ankle boots
- Warm short snow boots
- French-style wool beret

Input item 2, the French beret, is frequently included in outfits with:

- Lace-panel fitted dress
- Fleece casual hoodie
- Studded black leather lace-up shoes

GT candidate H is frequently included in outfits with:

- Slim gray straight jeans
- Black lace-panel leggings
- Knit beanie

Interpretation:

IB x BI does not copy the exact GT item from the context. Instead, it shows that the input bag and beret are directly connected to boot-like items in past outfits. This makes the ankle boots a strong item-level compatibility choice.

### IU x UI Additional Context

Input item 1, the black crossbody bag, is frequently co-purchased with:

- Similar small leather shoulder/crossbody bags
- Mini saddle/chain bags
- Striped sweatshirt

Input item 2, the French beret, is frequently co-purchased with:

- Black skinny jeans
- Women's watch
- Striped shirt

GT candidate H is frequently co-purchased with:

- Small floral earrings
- Nike running shoes
- Knit-lace dress

Interpretation:

IU x UI gives broader user-shopping preferences, but it does not focus as clearly on the boot slot for this input bundle.

### BI x IB Additional Context

No stronger retrieved outfit template than the IB x BI item-level signal. BI x IB ranked the GT third.

Interpretation:

The retrieved bundle-level information was less decisive than the direct item-level co-bundle links.

## Case 2: IU x UI Unique Success

Dataset: POG-Dense  
Bundle ID: 11363  
Successful view: IU x UI  
Ranks: IB x BI = 4, IU x UI = 1, BI x IB = 4

### Input

1. Small quilted crossbody bag
2. Loose striped sweatshirt

### Ground Truth

C. British-style black chunky leather shoes / school-style leather shoes

### Candidates

A. Irregular suspender dress  
B. Small Korean-style square crossbody bag  
C. British-style black chunky leather shoes  
D. Slim black jeans  
E. Men's loose turtleneck knit sweater  
F. Loose harem pants  
G. Wool lantern pants / wide tapered pants  
H. White letter-print T-shirt  
I. Loose striped T-shirt dress  
J. High-waist leather shorts

### IB x BI Additional Context

Input item 1, the quilted bag, is frequently included in outfits with:

- Black high-top canvas shoes
- Heeled winter ankle boots
- Retro flat ankle boots

Input item 2, the striped sweatshirt, is frequently included in outfits with:

- Wide-leg casual pants
- Black beret
- Snow boots

GT candidate C has no strong IB x BI context in this case.

Interpretation:

IB x BI sees some shoe-related compatibility around the input, but the candidate-level evidence is not strong enough. It ranks the GT fourth and prefers pants.

### IU x UI Additional Context

Input item 1, the quilted bag, is frequently co-purchased with:

- Platform white sneakers
- French-style mesh dress
- Simple leather handbag

Input item 2, the striped sweatshirt, is frequently co-purchased with:

- Short padded jacket
- Trendy white sneakers
- French-style slip dress

GT candidate C is frequently co-purchased with:

- Similar retro black leather shoes
- Sleeveless padded vest
- Short baseball jacket

Interpretation:

IU x UI does not contain the exact GT item as a copied context item. Instead, it reveals a user preference neighborhood around casual/student-style leather shoes and outerwear. This helps select the black leather shoes, while the other views move toward generic pants.

### BI x IB Additional Context

BI x IB did not retrieve an outfit template that clearly points to the leather-shoe slot. It ranked the GT fourth.

Interpretation:

The bundle-level template signal was weaker than the user co-purchase preference signal.

## Case 3: BI x IB Unique Success

Dataset: POG-Dense  
Bundle ID: 11208  
Successful view: BI x IB  
Ranks: IB x BI = 8, IU x UI = 9, BI x IB = 1

### Input

1. Floral chiffon dress with square neckline
2. Long pearl earrings

### Ground Truth

D. Steve Madden flat fur slippers / flat mule shoes

### Candidates

A. Oversized denim jacket  
B. Short faux-fur biker jacket  
C. Light denim jacket  
D. Steve Madden flat fur slippers / mule shoes  
E. Fleece slip-on shoes  
F. High-waist knit skirt  
G. Men's white sneakers  
H. Bohemian summer dress  
I. Short pink bodycon dress  
J. Large leather tote bag

### IB x BI Additional Context

Input item 1, the floral dress, is frequently included in outfits with:

- Nude pointed high heels
- Flat fashion slides
- Small leather handbag

Input item 2, the pearl earrings, is frequently included in outfits with:

- Vintage chain bag
- Trench-style dress
- Off-shoulder chiffon dress

GT candidate D is frequently included in outfits with:

- Long statement earrings
- Short printed hoodie

Interpretation:

IB x BI has some local item-level signals, but they are scattered. It incorrectly ranks a denim jacket higher than the GT.

### IU x UI Additional Context

Input item 1, the floral dress, is frequently co-purchased with:

- Wool shorts
- Preppy shirt dress
- French-style chiffon dress

Input item 2, the pearl earrings, is frequently co-purchased with:

- Black school-style loafers
- Chunky Oxford shoes
- Printed knit dress

GT candidate D has no strong IU x UI context in this case.

Interpretation:

IU x UI gives preference-level signals, but it does not clearly identify the missing shoe/slipper slot for this bundle. It ranks the GT ninth.

### BI x IB Additional Context

Retrieved past outfits:

1. Related outfit with the same floral dress and pearl earrings, plus a small crossbody bag and flat pointed shoes.
2. Related outfit with the same floral dress and pearl earrings, plus a small handbag and fur/slipper-style flat shoes.
3. Related outfit with the same floral dress and pearl earrings, plus fur/slipper-style flat shoes.

Interpretation:

BI x IB does not need the exact GT item to appear. The retrieved outfit templates repeatedly show that the dress-plus-earrings structure is commonly completed with flat shoes or slipper-like shoes. This makes the GT flat fur slippers the most natural missing slot.

## Slide-Level Takeaway

These three examples show that unique successes are not just exact item memorization:

- IB x BI captures fine-grained item-level co-bundle compatibility.
- IU x UI captures user preference and co-purchase neighborhoods.
- BI x IB captures retrieved outfit-level templates and missing-slot structure.

