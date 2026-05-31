# Unique-View Hit@1 Case Study

## Summary by Dataset and View
| dataset | unique_hit_view | count | avg_same_gt_category_candidates | avg_num_input_items | avg_distinct_input_categories | top_gt_categories | top_input_categories |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pog | BIxIB | 21 | 1.43 | 1.52 | 1.52 | Women's Chiffon and Floral Dresses (4); Women's Casual Shoes and Sneakers (3); Women's Mid-Length Woolen Coats (2); Women's High Waist Wide Leg Trousers (2); Men's Streetwear Hoodies and Sweatshirts (1) | Women's Fashion Shoulder and Crossbody Bags (8); Women's Casual Shoes and Sneakers (4); Women's Mid-Length Woolen Coats (2); Women's Heeled Ankle Boots (2); Women's Chiffon and Floral Dresses (2) |
| pog | IBxBI | 17 | 1.47 | 1.53 | 1.53 | Women's Heeled Ankle Boots (3); Women's Fashion Berets and Bucket Hats (3); Women's Casual Shoes and Sneakers (3); Men's Short Sleeve T-Shirts (1); Women's Loose Hoodies and Sweatshirts (1) | Women's Fashion Shoulder and Crossbody Bags (6); Women's Casual Shoes and Sneakers (3); Women's Fashion Berets and Bucket Hats (3); Women's Cropped Denim Jeans (3); Classic Canvas High and Low Top Sneakers (2) |
| pog | IUxUI | 14 | 1.71 | 1.71 | 1.71 | Women's Cropped Denim Jeans (3); Women's High-Waist Fashion Skirts (2); Women's Casual Shoes and Sneakers (2); Women's Fashion Shoulder and Crossbody Bags (1); Women's Mid-Length Woolen Coats (1) | Women's Fashion Shoulder and Crossbody Bags (4); Women's Casual Shoes and Sneakers (3); Women's Mid-Length Woolen Coats (2); Women's Fleece-Lined Thermal Leggings (2); Sterling Silver Fashion Earrings (1) |
| pog_dense | BIxIB | 27 | 1.48 | 2.07 | 2.07 | Women's Leather Crossbody Bags (4); Women's Dress Pumps and Heels (3); Women's Fashion Sun Hats (3); Women's High-Waist Mid-Length Skirts (2); Women's Fashion Ankle Boots (2) | Women's Leather Crossbody Bags (12); Women's Dress Pumps and Heels (10); Women's Chiffon Summer Dresses (7); Women's Fashion Drop Earrings (5); Women's Fashion Ankle Boots (5) |
| pog_dense | IBxBI | 33 | 1.55 | 2.00 | 2.00 | Women's Leather Crossbody Bags (9); Women's Dress Pumps and Heels (5); Women's Fashion Sun Hats (3); Women's Ankle Strap Heeled Sandals (2); Women's Sterling Silver Stud Earrings (2) | Women's Leather Crossbody Bags (11); Women's Dress Pumps and Heels (11); Women's Fashion Drop Earrings (7); Women's Fashion Sun Hats (7); Women's Chiffon Summer Dresses (6) |
| pog_dense | IUxUI | 13 | 1.92 | 2.00 | 2.00 | Women's Leather Crossbody Bags (3); Women's Fashion Sun Hats (2); Women's Dress Pumps and Heels (2); Women's Chiffon Summer Dresses (2); Women's High Waist Casual Trousers (1) | Women's Leather Crossbody Bags (6); Women's Fashion Ankle Boots (4); Women's Dress Pumps and Heels (4); Women's Chiffon Summer Dresses (3); Women's Korean Style Hoodies and Sweatshirts (2) |

## Top GT Categories
| dataset | unique_hit_view | true_category_name | count | ratio_within_view |
| --- | --- | --- | --- | --- |
| pog | BIxIB | Women's Chiffon and Floral Dresses | 4 | 0.19 |
| pog | BIxIB | Women's Casual Shoes and Sneakers | 3 | 0.14 |
| pog | BIxIB | Women's Mid-Length Woolen Coats | 2 | 0.10 |
| pog | BIxIB | Women's High Waist Wide Leg Trousers | 2 | 0.10 |
| pog | BIxIB | Men's Streetwear Hoodies and Sweatshirts | 1 | 0.05 |
| pog | IBxBI | Women's Heeled Ankle Boots | 3 | 0.18 |
| pog | IBxBI | Women's Fashion Berets and Bucket Hats | 3 | 0.18 |
| pog | IBxBI | Women's Casual Shoes and Sneakers | 3 | 0.18 |
| pog | IBxBI | Men's Short Sleeve T-Shirts | 1 | 0.06 |
| pog | IBxBI | Women's Loose Hoodies and Sweatshirts | 1 | 0.06 |
| pog | IUxUI | Women's Cropped Denim Jeans | 3 | 0.21 |
| pog | IUxUI | Women's High-Waist Fashion Skirts | 2 | 0.14 |
| pog | IUxUI | Women's Casual Shoes and Sneakers | 2 | 0.14 |
| pog | IUxUI | Women's Fashion Shoulder and Crossbody Bags | 1 | 0.07 |
| pog | IUxUI | Women's Mid-Length Woolen Coats | 1 | 0.07 |
| pog_dense | BIxIB | Women's Leather Crossbody Bags | 4 | 0.15 |
| pog_dense | BIxIB | Women's Dress Pumps and Heels | 3 | 0.11 |
| pog_dense | BIxIB | Women's Fashion Sun Hats | 3 | 0.11 |
| pog_dense | BIxIB | Women's High-Waist Mid-Length Skirts | 2 | 0.07 |
| pog_dense | BIxIB | Women's Fashion Ankle Boots | 2 | 0.07 |
| pog_dense | IBxBI | Women's Leather Crossbody Bags | 9 | 0.27 |
| pog_dense | IBxBI | Women's Dress Pumps and Heels | 5 | 0.15 |
| pog_dense | IBxBI | Women's Fashion Sun Hats | 3 | 0.09 |
| pog_dense | IBxBI | Women's Ankle Strap Heeled Sandals | 2 | 0.06 |
| pog_dense | IBxBI | Women's Sterling Silver Stud Earrings | 2 | 0.06 |
| pog_dense | IUxUI | Women's Leather Crossbody Bags | 3 | 0.23 |
| pog_dense | IUxUI | Women's Fashion Sun Hats | 2 | 0.15 |
| pog_dense | IUxUI | Women's Dress Pumps and Heels | 2 | 0.15 |
| pog_dense | IUxUI | Women's Chiffon Summer Dresses | 2 | 0.15 |
| pog_dense | IUxUI | Women's High Waist Casual Trousers | 1 | 0.08 |

## Top Input-to-GT Category Pairs
| dataset | unique_hit_view | input_category_name | true_category_name | count |
| --- | --- | --- | --- | --- |
| pog | BIxIB | Women's Fashion Shoulder and Crossbody Bags | Women's Casual Shoes and Sneakers | 2 |
| pog | BIxIB | Women's Fashion Shoulder and Crossbody Bags | Women's Chiffon and Floral Dresses | 2 |
| pog | BIxIB | Classic Branded Sneakers and Skate Shoes | Men's Streetwear Hoodies and Sweatshirts | 1 |
| pog | BIxIB | Women's Long-Sleeved Blouses and Shirts | Women's Fleece-Lined Thermal Leggings | 1 |
| pog | BIxIB | Women's Mid-Length Woolen Coats | Women's Fleece-Lined Thermal Leggings | 1 |
| pog | IBxBI | Women's Fashion Shoulder and Crossbody Bags | Women's Heeled Ankle Boots | 3 |
| pog | IBxBI | Women's Fashion Shoulder and Crossbody Bags | Women's Casual Shoes and Sneakers | 2 |
| pog | IBxBI | Classic Canvas High and Low Top Sneakers | Men's Short Sleeve T-Shirts | 1 |
| pog | IBxBI | Women's Casual Shoes and Sneakers | Women's Loose Hoodies and Sweatshirts | 1 |
| pog | IBxBI | Women's Fashion Berets and Bucket Hats | Women's Heeled Ankle Boots | 1 |
| pog | IUxUI | Sterling Silver Fashion Earrings | Women's High-Waist Fashion Skirts | 1 |
| pog | IUxUI | Women's High Waist Wide Leg Trousers | Women's Fashion Shoulder and Crossbody Bags | 1 |
| pog | IUxUI | Women's Casual Shoes and Sneakers | Women's Fashion Shoulder and Crossbody Bags | 1 |
| pog | IUxUI | Women's Heeled Ankle Boots | Women's Cropped Denim Jeans | 1 |
| pog | IUxUI | Women's Casual Shoes and Sneakers | Women's Mid-Length Woolen Coats | 1 |
| pog_dense | BIxIB | Women's Leather Crossbody Bags | Women's Dress Pumps and Heels | 3 |
| pog_dense | BIxIB | Women's Chiffon Summer Dresses | Women's Leather Crossbody Bags | 3 |
| pog_dense | BIxIB | Women's Dress Pumps and Heels | Women's Leather Crossbody Bags | 3 |
| pog_dense | BIxIB | Women's Fashion Drop Earrings | Women's High-Waist Mid-Length Skirts | 2 |
| pog_dense | BIxIB | Women's Chiffon Blouses and Shirts | Women's High-Waist Mid-Length Skirts | 2 |
| pog_dense | IBxBI | Women's Dress Pumps and Heels | Women's Leather Crossbody Bags | 5 |
| pog_dense | IBxBI | Women's Leather Crossbody Bags | Women's Dress Pumps and Heels | 4 |
| pog_dense | IBxBI | Women's Fashion Drop Earrings | Women's Leather Crossbody Bags | 3 |
| pog_dense | IBxBI | Women's Fashion Sun Hats | Women's Leather Crossbody Bags | 2 |
| pog_dense | IBxBI | Women's Fashion Drop Earrings | Women's Ankle Strap Heeled Sandals | 2 |
| pog_dense | IUxUI | Women's Leather Crossbody Bags | Women's Dress Pumps and Heels | 2 |
| pog_dense | IUxUI | Women's Dress Pumps and Heels | Women's Leather Crossbody Bags | 2 |
| pog_dense | IUxUI | Women's Chiffon Summer Dresses | Women's Fashion Sun Hats | 1 |
| pog_dense | IUxUI | Women's Fashion Ankle Boots | Women's Fashion Sun Hats | 1 |
| pog_dense | IUxUI | Women's Leather Crossbody Bags | Women's High Waist Casual Trousers | 1 |