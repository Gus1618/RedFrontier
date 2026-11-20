# RedFrontier — Finding a Piece of the Moon in South America

Finding a piece of the Moon on Earth with NASA satellites comes with an unusual challenge: you have to make the computer ignore everything that makes Earth act like Earth—our thick atmosphere, liquid water, weather systems, and vegetation.

Instead of picking planetary test sites because they look like a sci-fi desert in photos, this repository uses open NASA Earth observation datasets (**SMAP**, **VIIRS**, **OMPS**, and **EMIT**) processed on a GPU to scan entire continents under the exact same physical ruler. When we let the orbital data speak for itself across South America, the algorithm pointed to an overlooked region high in the Bolivian Andes: **Challapata, in the department of Oruro**.

This README walks through how to set up and run the pipeline, the physics behind every satellite variable we track, what the resulting heatmaps reveal about Oruro, a step-by-step breakdown of how the lunar similarity score is calculated, and an honest look at how well Challapata actually mirrors the lunar surface.

---

## 1. Prerequisites

Before running the notebook (`FindAnalogSpots.ipynb`), you need two things ready: a GPU environment and an authorized NASA Earthdata account.

### 1.1 Hardware Requirement: A CUDA-Capable GPU
Processing continental and global satellite rasters across multiple NASA missions involves millions of pixels stacked across time and spectral bands. Doing that matrix math on a standard CPU is painfully slow, so the pipeline offloads heavy grid operations to a GPU using **CuPy** alongside `xarray`, `dask`, and `earthaccess`.

* **Local machine:** An NVIDIA GPU with CUDA drivers installed.
* **Cloud (recommended):** Google Colab with the runtime switched to **T4 GPU** (or better) via `Runtime > Change runtime type > Hardware accelerator > GPU`.

### 1.2 Authorizing the `NASA GESDISC DATA ARCHIVE` App
The code streams calibrated scientific files (NetCDF/HDF5) directly from NASA's cloud archives using the `earthaccess` Python library. A free account is required, and you **must** pre-authorize the GES DISC archive so command-line and Python tools are allowed to pull data on your behalf.

1. Create a free account (or log in) at **[NASA Earthdata Login](https://urs.earthdata.nasa.gov/)**.
2. In the top navigation bar, click **Applications**, then select **Authorized Apps**.
3. Scroll to the bottom of the page and click **APPROVE MORE APPLICATIONS**.
4. In the search bar, type **`NASA GESDISC DATA ARCHIVE`** and click **Search**.
5. Select the application, click **Approve**, and accept the End User License Agreement (EULA).
6. Once it appears in your list of approved applications, your account is ready.

> **Important:** If you are running the code locally rather than interactively in Colab via `earthaccess.login()`, store your credentials in a `.netrc` file in your home directory (`~/.netrc` on Linux/macOS or `C:\Users\<username>\.netrc` on Windows):
> ```text
> machine urs.earthdata.nasa.gov login <your_username> password <your_password>
> ```

---

## 2. The Science Behind the Criteria: Three Physical Pillars

To make this work for both a high school student and a planetary scientist, think of the algorithm as a filter for **hardware failure modes**. When space agencies test lunar rovers, solar arrays, drills, or habitats on Earth, they aren't looking for pretty scenery—they are looking for places that stress metal, batteries, optics, and electronics the way the Moon does.

We organized four NASA orbital instruments into **three physical pillars**:

### Pillar 1 — Temperature & Albedo (Thermal Shock & Light Absorption)
* **NASA Instruments:** VIIRS & SMAP (`surface_temp` / `LST_1KM`, `Albedo_BSA_shortwave`)
* **The intuition:** On Earth, the atmosphere acts like a blanket, keeping daytime heat from spiking too high and trapping warmth at night. The Moon has no blanket—it sits in vacuum, swinging by more than **$250\text{ K}$** between sunlit plains ($> 120^\circ\text{C}$) and deep shadow ($< -130^\circ\text{C}$). On top of that, the dark plains of the Moon (the basaltic *maria*) are physically dark, absorbing most of the sunlight that hits them instead of reflecting it away.
* **The physics:** We measure **diurnal thermal amplitude ($\Delta T = T_{\text{day}} - T_{\text{night}}$)** and **shortwave Black-Sky Albedo (BSA)**. High $\Delta T$ exposes mechanical joints, adhesives, and lithium batteries to rapid expansion-contraction fatigue. Low albedo ($\le 0.12\text{–}0.15$) mimics the optical reflectance and radiative heat absorption of lunar basalt—something bright white quartz sand dunes or blinding salt crusts fail to reproduce.

### Pillar 2 — Soil Desiccation & Biological Sterility (Zero Water, Zero Life)
* **NASA Instruments:** SMAP & VIIRS (`sm_surface`, `sm_rootzone`, `sm_profile`, `land_fraction_wilting`, `NDVI`)
* **The intuition:** Earth is covered in water and plants; the Moon is bone-dry, sterile rock dust (regolith) with zero liquid water and zero chlorophyll.
* **The physics:** Even small amounts of liquid water or organic matter ruin a lunar field test. Water has a huge specific heat capacity (it buffers temperature swings), changes the soil's **dielectric permittivity** (throwing off ground-penetrating radar and water-prospecting sensors), and glues loose grains together mechanically. By combining volumetric soil moisture across three depths—surface ($0\text{–}5\text{ cm}$), root zone ($0\text{–}100\text{ cm}$), and deep profile—with the **permanent wilting point fraction** (`land_fraction_wilting`, measuring chronic aridity) and the **Normalized Difference Vegetation Index (NDVI)**, the model strips away forests, farms, and damp basins to leave only sterile, exposed substrates.

### Pillar 3 — Atmospheric Stability & Mafic Mineralogy (Calm Skies & Iron-Rich Rock)
* **NASA Instruments:** OMPS, VIIRS AERDB & EMIT (`aerosol_idx`, `AOD`, `group_2_band_depth`)
* **The intuition:** Unlike Mars, which gets hit by planet-wide dust storms, the Moon has no wind. Its skies are completely clear, and its dark plains are rich in volcanic rocks loaded with iron and magnesium.
* **The physics:** We use the **OMPS UV Aerosol Index** and **Aerosol Optical Depth (AOD at $550\text{ nm}$)** to *penalize* dusty, turbulent basins and reward clear, stable skies that emulate the windless lunar exosphere. At the same time, we pull hyperspectral mineral data from **EMIT** (an imaging spectrometer mounted on the International Space Station). Specifically, `group_2_band_depth` tracks spectral absorption features tied to **iron- and magnesium-rich mafic silicates**—the exact geochemical family that makes up lunar basaltic regolith.

---

## 3. Map Analysis: Why Oruro Stands Out in South America

When you run the model across South America, the visual output immediately challenges the conventional assumption that the Atacama Desert is the only game in town.

### Regional View: The Andean Lunar Corridor
![Top 5% Lunar Analog Heatmap](mapa_mundi_zoom_1.png)

Looking at the **Top 5% Lunar Analog Heatmap**, almost the entire continent drops out of contention—the Amazon, the Cerrado, the Pampas, and coastal plains are filtered away by moisture and vegetation (with only faint, isolated smudges appearing in drier pockets of Brazil). What lights up in bright yellow is a continuous high-altitude belt running down the central Andes, spanning southern Peru, northern Chile, northwestern Argentina, and—most strikingly—**western Bolivia**.

### Local View: Zooming into Oruro and the Lake Poopó Basin
![Challapata and Oruro Top 10 Cluster](mapa_mundi_zoom_2.png)

Zooming straight into Oruro brings us to the desiccated margins of the **Lake Poopó basin** in the department of **Oruro, Bolivia**, sitting at more than **$3,700\text{ meters}$** above sea level.

What makes this map remarkable isn't just a single lucky pixel—it is the **density** of top-ranking spots across the exact same department. Alongside our #1 site in Challapata (`0.676`), neighboring coordinates across Challapata (#3 at `0.674`, #5 at `0.667`, #7 at `0.656`, #10 at `0.634`) and Totoral (#8 at `0.644`) dominate the regional rankings. That spatial consistency tells us we aren't looking at sensor noise; we are looking at a broad, geophysically coherent proving ground.

### Oruro's Potential for Local Space Research
Recognizing Oruro as a top-tier lunar analog isn't just an academic curiosity—it changes the practical economics of aerospace research in South America:

* **A low-cost natural laboratory:** Field-testing rovers, thermal insulation, solar panels, and telemetry nodes normally requires expensive expeditions to the Arctic, Antarctica, or the Northern Hemisphere. Having a high-altitude analog right in the Andes removes a massive logistical barrier for South American universities and startups.
* **Expanding the Agencia Boliviana Espacial (ABE):** Historically, Bolivia's space agency has focused on telecommunications and Earth-orbiting satellites like *Túpac Katari*. Identifying world-class planetary analog terrain inside Bolivian territory gives ABE and local universities a clear path to evolve from technology users into active protagonists in space hardware validation, geospatial software, and instrumentation.
* **International partnerships & talent:** Positioning Oruro on the global analog map creates a direct bridge for joint field campaigns with **NASA's Artemis Program** and the **European Space Agency (ESA)**—bringing scientific infrastructure and hands-on engineering challenges directly to local students.

---

## 4. The Result: Challapata, Oruro, Bolivia (#1 in South America)

After scanning the continent, the algorithm ranked **Challapata, Oruro, Bolivia** (`-19.0148° N, -66.9865° W`) as the **#1 lunar analog in South America**, achieving a final similarity score of **`0.6760`**.

Here is the exact NASA orbital telemetry recorded at the winning coordinate:

| Pillar & Geophysical Metric | NASA Sensor / Product | Measured Value at Challapata | What It Means for Lunar Simulation |
| :--- | :--- | :--- | :--- |
| **Diurnal Thermal Amplitude ($\Delta T$)** | VIIRS / SMAP (LST) | **`41.93 K`** | Thin, dry Altiplano air drives intense daytime solar heating and rapid nighttime radiative cooling. |
| **Shortwave Albedo (BSA)** | VIIRS / MODIS | **`0.1500`** | Dark volcanic-sedimentary surface absorbs solar radiation similarly to basaltic lunar *maria*. |
| **Surface Soil Moisture ($0\text{–}5\text{ cm}$)** | SMAP (`sm_surface`) | **`0.05864 m³/m³`** | Severely desiccated top layer minimizes thermal inertia distortion and electrical conductivity. |
| **Root Zone Moisture ($0\text{–}100\text{ cm}$)** | SMAP (`sm_rootzone`) | **`0.11150 m³/m³`** | Dry, stable subsurface profile suited for drill testing and foundation anchoring. |
| **Deep Profile Moisture** | SMAP (`sm_profile`) | **`0.11473 m³/m³`** | Consistently low water content deep below the surface. |
| **Wilting Point Fraction (Aridity)** | SMAP L4 (`wilting`) | **`0.8682`** | Extreme pedological aridity across the sandy/saline plains bordering the Lake Poopó basin. |
| **Vegetation Index (NDVI)** | VIIRS (`NDVI`) | **`0.2000`** | Near-sterile terrain with minimal biomass interfering with regolith trials. |
| **UV Aerosol Index (UVAI)** | OMPS | **`0.2236`** | Very low suspended dust loading in the atmosphere. |
| **Aerosol Optical Depth (AOD)** | VIIRS AERDB | **`0.1220`** | Clean, transparent sky emulating the calm, windless lunar exosphere. |
| **Mafic Mineral Abundance** | EMIT (`group_2_band_depth`) | **`0.3138`** | Strong iron- and magnesium-rich spectral signature matching basaltic lunar crust chemistry. |

---

## 5. Step-by-Step: How the Score Is Calculated

This is the core of the code. Turning raw satellite gigabytes into a single number between `0` and `1` requires solving a fundamental Earth-science dilemma: **no single pixel on Earth maximizes every lunar property at once.**

For example, Brazil has some of the purest basaltic rock formations on the planet—but they are wet and covered in tropical forest. Meanwhile, the driest salt flats on Earth are bone-dry—but their bright white crusts reflect sunlight like a mirror instead of absorbing it like dark lunar rock. If you just take a simple average of all variables, a wet volcanic jungle or a blinding white salt flat can cheat its way to a decent score.

Here is how our pipeline solves that, step by step:

### Step 1: Cloud Streaming & Spatial Grid Alignment
Using `earthaccess`, the script queries NASA's DAACs for SMAP, VIIRS, OMPS, and EMIT granules over the target bounding box and streams them directly into memory with `xarray` and `dask`. Because each satellite sees the Earth at a different native resolution (e.g., SMAP at 9 km, VIIRS at 1 km, OMPS at ~10–12 km), all layers are reprojected and resampled onto a single common latitude/longitude grid (`EPSG:4326`), then transferred to GPU memory as aligned **CuPy** arrays.

### Step 2: Normalization & Directional Inversion ($0 \to 1$)
Raw geophysical units can't be added directly—you can't add Kelvin to cubic meters of water per cubic meter of soil. Every variable $x$ is normalized to a dimensionless scale from $0.0$ (worst match) to $1.0$ (ideal lunar match), respecting its physical direction:

* **Direct scaling (higher is more lunar):** Applied to Diurnal Thermal Amplitude ($\Delta T$), Wilting Point Fraction (Aridity), and EMIT Mafic Mineral Abundance (`group_2_band_depth`).
* **Inverted scaling (lower is more lunar, $1 - x_{\text{norm}}$):** Applied to Soil Moisture (`sm_surface`, `sm_rootzone`, `sm_profile`), Vegetation (`NDVI`), Shortwave Albedo (`BSA`), UV Aerosol Index (`UVAI`), and Aerosol Optical Depth (`AOD`).

### Step 3: Pillar Aggregation
The normalized variables are grouped into three sub-scores representing our three physical pillars:
* **$S_{\text{solo}}$ (Pillar 2 — Soil & Desiccation):** Combines inverted surface/root-zone/deep moisture, aridity (`wilting`), and inverted `NDVI`.
* **$S_{\text{term}}$ (Pillar 1 — Thermodynamics & Albedo):** Combines high diurnal thermal swing ($\Delta T$) and low shortwave albedo.
* **$S_{\text{min}}$ (Pillar 3 — Atmosphere & Mineralogy):** Combines inverted aerosols (`UVAI`, `AOD`) with EMIT mafic mineral depth.

### Step 4: Non-Compensatory Hierarchical Prioritization
Instead of a naive average, the GPU computes the final lunar similarity score using a **non-compensatory hierarchical objective function**:

$$\text{Score}_{\text{Lua}} = \left( W_{\text{solo}} \cdot S_{\text{solo}} + W_{\text{term}} \cdot S_{\text{term}} + W_{\text{min}} \cdot S_{\text{min}} \right) \times \Phi_{\text{inibidor}}$$

Think of this equation as a three-stage engineering gauntlet:

1. **Level 1 — Eliminatory Baseline Filter ($\Phi_{\text{inibidor}}$ & Highest Weight $W_{\text{solo}}$):** Soil dryness and absence of vegetation act as the gatekeeper. If a pixel is wet or vegetated, the inhibitor term $\Phi_{\text{inibidor}}$ crushes its final score—regardless of how basaltic its rocks are. Without a dry, bare substrate, thermal inertia and radar tests are physically invalid.
2. **Level 2 — High-Altitude Thermodynamic Stress ($W_{\text{term}}$):** Once a pixel passes the aridity gate, the model heavily rewards extreme day-to-night temperature swings ($\Delta T > 40\text{ K}$) paired with dark albedo. On Earth, radiative heat loss that rapid only happens where the air column is thin and dry, which naturally pulls the algorithm toward plateaus above $3,700\text{ meters}$.
3. **Level 3 — Optical & Geochemical Tie-Breaker ($W_{\text{min}}$):** Among equally dry, high-altitude basins, Pillar 3 breaks the tie by rewarding clear, dust-free skies (low AOD) and strong iron/magnesium mafic signatures from EMIT.

**How Challapata reaches `0.6760`:** Challapata passes the $\Phi_{\text{inibidor}}$ gate cleanly thanks to its high aridity (`0.8682`), very dry topsoil (`0.05864 m³/m³`), and low biomass (`NDVI = 0.2000`). It then scores near the terrestrial ceiling in Pillar 1 with its brutal `41.93 K` daily thermal swing and dark `0.1500` albedo, and wins the tie-breaker in Pillar 3 with clean skies (`AOD = 0.1220`) and a strong mafic rock signature (`0.3138`). No terrestrial spot can hit `1.0`—because Earth still has air, residual moisture, and weathered soil—making `0.6760` an exceptionally high score for a real-world site.

### Step 5: Spatial Filtering (80 km NMS) & Interactive Mapping
If you just ask a computer for the "Top 10 pixels," it will give you 10 pixels sitting side-by-side in the same valley. To separate distinct geographical basins when generating regional or global rankings, the pipeline implements **Spatial Non-Maximum Suppression (NMS)** using the **Haversine distance formula** with an **$80\text{ km}$ radius**. Finally, the results are matched to real place names via offline reverse geocoding and rendered into interactive HTML maps using **Folium** (with ESRI Satellite Imagery and CartoDB Dark Matter layers).

---

## 6. How Well Does Challapata Actually Fit the Moon?

Good science means being honest about both what works and where the analogy breaks down.

### Where Challapata Excels
* **Thermal-cycle testing:** A `41.93 K` daily swing at $>3,700\text{ m}$ altitude provides realistic thermal shock for validating battery management systems, solar cell efficiency, microcontrollers, and structural joints.
* **Optical & spectral calibration:** Its `0.1500` albedo and `0.3138` EMIT mafic signature offer a dark, iron-rich baseline for calibrating rover cameras, spectrometers, and autonomous navigation under high-contrast high-altitude sunlight.
* **Subsurface & dielectric realism:** The low moisture profile (`0.05864 m³/m³` surface, `0.8682` wilting fraction) keeps thermal inertia and electrical conductivity low enough for meaningful ground-penetrating radar (GPR) and anchoring trials.

### Where Earth Cannot Imitate the Moon (Physical Limits & Mitigations)
No place on Earth is 100% lunar. Engineers running field campaigns in Challapata must still account for four physical boundaries through mathematical scaling or lab chambers:

| Physical Domain | Challapata Field Condition | Real Lunar Surface (*Maria*) | Engineering Impact & Mitigation |
| :--- | :--- | :--- | :--- |
| **Pressure & Convection** | $\approx 64\text{ kPa}$ ($0.63\text{ atm}$) + wind | Exospheric vacuum ($\approx 3 \times 10^{-15}\text{ atm}$) | Thin Altiplano air still cools hardware via convection. Thermal models validated in the field must be recalculated with convective heat transfer set to zero ($h_c \to 0$), relying strictly on conduction and radiation ($\propto T^4$). |
| **Dust Morphology** | Grains rounded by wind and water | Jagged, glassy, impact-shattered *agglutinates* | Terrestrial sand underestimates mechanical wear on rotary seals and gears. Critical mechanisms still need chamber testing with sharp simulants (e.g., **LMS-1** or **JSC-1A**). |
| **Electrostatics & Moisture** | Residual moisture ($\approx 5.86\%$) dissipates charge | Absolute dryness ($0\%$) + direct solar wind | Lunar dust charges up electrostatically, levitating and clinging aggressively to lenses and radiators. Adhesion observed in Oruro is only a lower bound. |
| **Gravity & Radiation** | $9.81\text{ m/s}^2$ ($1.0\text{ g}$) + magnetosphere | $1.62\text{ m/s}^2$ ($0.166\text{ g}$) + cosmic rays (GCR) | Rover traction and wheel-sinkage tests require $1/6$ mass-scaling compensation, and avionics still require particle-beam radiation testing. |

---

## 7. Why This Discovery Matters

For decades, planetary analog research has gravitated toward the same handful of famous sites—the Atacama Desert, Utah, Hawaii, Iceland, Antarctica—simply because they were already in the literature. Meanwhile, vast stretches of the Southern Hemisphere went unscreened under a unified, quantitative methodology.

Without an open, data-driven pipeline like this, the scientific potential of **Oruro and Challapata** could easily have gone undiscovered forever. That is what makes initiatives like the **NASA Space Apps Challenge** so consequential: by putting NASA's open orbital archives in the hands of students, engineers, and curious problem-solvers worldwide, a weekend hackathon can uncover a real-world proving ground in the Bolivian Altiplano—one capable of turning a region into a hub for South American space exploration and helping prepare the hardware that will take humanity back to the Moon.
