import React from "react";

// Vite: grab every svg in src/assets/plots as a URL
const svgs = Object.values(
  import.meta.glob("./assets/plots/*.svg", { eager: true, as: "url" })
);

export default function PlotsGrid() {
  return (
    <div style={styles.grid}>
      {svgs.map((src) => (
        <div key={src} style={styles.card}>
          <img src={src} alt="" style={styles.img} />
        </div>
      ))}
    </div>
  );
}

const styles = {
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))", // always 2 columns
    gap: 16,
    padding: 16,
    maxWidth: 1300,          // optional: keeps it from getting too wide
    margin: "0 auto",        // centers the whole grid
    alignItems: "start",
  },
  card: {
    border: "1px solid rgba(0,0,0,0.12)",
    borderRadius: 12,
    padding: 10,
    overflow: "hidden",      // keeps rounded corners clean
  },
  img: {
    width: "100%",
    height: "auto",          // preserves aspect ratio
    display: "block",
  },
};
