# 2-DoF Planar Robot Kolu – Dinamik Modelleme ve Ters Dinamik Kontrol Simülasyonu

**Ders:** YZR502u04a01 | **Ödev:** ÖDEV0401  
**YouTube (Sunum):** https://youtu.be/6IXRZU21WQg

---

## İçerik

```
robot_2dof/
├── robot_dynamics.py   # Dinamik model: M(q), C(q,q̇), G(q), ters kinematik, Jacobian
├── trajectory.py       # Dairesel ve kare yörünge üreticileri
├── simulation.py       # Ana simülasyon betiği (argüman destekli)
├── figures/            # Üretilen grafik dosyaları
│   ├── 01_joint_positions.png
│   ├── 02_joint_errors.png
│   ├── 03_joint_torques.png
│   ├── 04_cartesian_trajectory.png
│   └── 05_gain_analysis.png
└── README.md
```

---

## Gereksinimler

Python 3.9 veya üzeri.

```bash
pip install numpy scipy matplotlib
```

---

## Kullanım

### Temel simülasyon (dairesel yörünge, varsayılan kazançlar)

```bash
cd robot_2dof
python simulation.py
```

### Özel parametrelerle çalıştırma

```bash
python simulation.py --traj circular --Kp 100 --Kv 20 --duration 12
python simulation.py --traj square   --Kp 200 --Kv 40 --duration 8
```

| Argüman      | Açıklama                          | Varsayılan |
|--------------|-----------------------------------|------------|
| `--traj`     | `circular` veya `square`          | `circular` |
| `--Kp`       | Pozisyon kazancı (köşegen)        | `100.0`    |
| `--Kv`       | Hız kazancı (köşegen)             | `20.0`     |
| `--duration` | Simülasyon süresi (saniye)        | `12.0`     |

Çalıştırmanın ardından `figures/` klasöründe grafikler oluşturulur; konsola RMS takip hatası, maksimum hata ve maksimum tork değerleri yazdırılır.

---

## Robot Parametreleri

| Parametre                 | Sembol | Değer        |
|---------------------------|--------|--------------|
| Bağlantı 1 uzunluğu      | L1     | 0.5 m        |
| Bağlantı 2 uzunluğu      | L2     | 0.5 m        |
| Bağlantı 1 kütlesi       | m1     | 1.0 kg       |
| Bağlantı 2 kütlesi       | m2     | 0.5 kg       |
| Bağlantı 1 eylemsizliği  | I1     | 0.0208 kg.m² |
| Bağlantı 2 eylemsizliği  | I2     | 0.0052 kg.m² |
| Yerçekimi ivmesi          | g      | 9.81 m/s²    |

---

## Dinamik Model (Özet)

Hareket denklemleri Lagrange yöntemiyle türetilmiştir:

    M(q).q'' + C(q,q').q' + G(q) = tau

**Kütle matrisi M(q)**

    M11 = m1*lc1^2 + I1 + m2*(L1^2 + lc2^2 + 2*L1*lc2*cos(q2)) + I2
    M12 = m2*(lc2^2 + L1*lc2*cos(q2)) + I2
    M22 = m2*lc2^2 + I2

**Coriolis matrisi C(q, q')**

    h   = m2*L1*lc2*sin(q2)
    C   = [[-h*q2',  -h*(q1'+q2')],
           [ h*q1',   0          ]]

**Yerçekimi vektörü G(q)**

    G1 = (m1*lc1 + m2*L1)*g*cos(q1) + m2*lc2*g*cos(q1+q2)
    G2 = m2*lc2*g*cos(q1+q2)

---

## Kontrolör (Ters Dinamik + PD)

    tau = M(q)*[q''_d + Kv*(q'_d - q') + Kp*(q_d - q)] + C(q,q')*q' + G(q)

Kapalı çevrimde hata dinamiği doğrusal hale gelir:

    e'' + Kv*e' + Kp*e = 0

---

## Simülasyon Sonuçları (Kp=100, Kv=20, Dairesel Yörünge)

| Metrik               | Eklem 1   | Eklem 2   |
|----------------------|-----------|-----------|
| RMS Takip Hatası     | 0.077°    | 0.049°    |
| Maks. Takip Hatası   | 0.619°    | 0.397°    |
| Maks. Tork           | 6.37 N.m  | 0.54 N.m  |

---

## Dosya Açıklamaları

### `robot_dynamics.py`
- `mass_matrix(q)` — M(q) hesabı
- `coriolis_matrix(q, dq)` — C(q, q') hesabı
- `gravity_vector(q)` — G(q) hesabı
- `inverse_dynamics_control(...)` — tau hesabı
- `forward_kinematics(q)` — uç-efektör konumu
- `inverse_kinematics(x, y)` — eklem açıları
- `jacobian(q)` ve `jacobian_dot(q, dq)`
- `robot_ode(t, state, ...)` — ODE sağ tarafı (scipy için)

### `trajectory.py`
- `circular_trajectory(t, ...)` — Dairesel yörünge (analitik türevli)
- `square_trajectory(t, ...)` — Kare yörünge (Fourier yaklaşımı, C² sürekli)

Her fonksiyon `(qd, dqd, ddqd)` üçlüsünü döndürür.

### `simulation.py`
- `run_simulation(...)` — Kapalı çevrim RK45 entegrasyonu, tüm verileri döndürür
- `plot_joint_positions(...)` — Eklem pozisyon grafikleri
- `plot_joint_errors(...)` — Takip hatası grafikleri
- `plot_torques(...)` — Tork zaman grafikleri
- `plot_cartesian(...)` — Kartezyen yörünge (x-y düzlemi)
- `plot_gain_analysis(...)` — Farklı kazançlar için RMS hata çubuğu

---

## Kaynaklar

1. Spong MW, Hutchinson S, Vidyasagar M. *Robot Modeling and Control.* 2. baskı. Wiley; 2020.
2. Siciliano B, Sciavicco L, Villani L, Oriolo G. *Robotics: Modelling, Planning and Control.* Springer; 2009. DOI: 10.1007/978-1-84628-642-1
3. Craig JJ. *Introduction to Robotics: Mechanics and Control.* 3. baskı. Pearson Prentice Hall; 2005.
4. Slotine JJE, Li W. *Applied Nonlinear Control.* Prentice Hall; 1991.
5. Dhaouadi R, Abu Hatab A. Dynamic modelling of differential-drive mobile robots using Lagrange and Newton-Euler methodologies. *Advances in Robotics & Automation* 2013; 2(2): 1-7.
6. Khalil W, Dombre E. *Modeling, Identification and Control of Robots.* Butterworth-Heinemann; 2004.
