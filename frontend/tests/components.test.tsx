import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { BandViewSwitcher } from "@/components/ui/BandViewSwitcher";

describe("StatusBadge Component", () => {
  it("renders with given status text and variant", () => {
    render(<StatusBadge status="Ready" variant="success" />);
    expect(screen.getByText("Ready")).toBeDefined();
  });
});

describe("BandViewSwitcher Component", () => {
  it("calls onBandChange when a band button is clicked", () => {
    let selected = "rgb";
    const handleChange = (band: any) => {
      selected = band;
    };

    render(
      <BandViewSwitcher
        activeBand="rgb"
        onBandChange={handleChange}
        availableBands={["rgb", "cir", "ndvi"]}
      />
    );

    const cirButton = screen.getByText("CIR");
    expect(cirButton).toBeDefined();
    fireEvent.click(cirButton);
    expect(selected).toBe("cir");
  });
});
