# Theme Configuration

This project uses **Tailwind CSS** for styling, with a customized configuration to enforce a unified light theme.

## Color Palette

The theme is built around the **Slate** color scale (bluish-grey) to provide a professional, clean look for the risk control system.

### Core Colors
| Token | Hex | Usage |
|-------|-----|-------|
| `primary` | `#3b82f6` (Blue-500) | Main brand color, active states, primary buttons |
| `accent` | `#60a5fa` (Blue-400) | Secondary highlights |
| `protected` | `#f59e0b` (Amber-500) | Warnings, protected fields |

### Backgrounds & Surfaces (Light Mode)
| Token | Hex | Usage |
|-------|-----|-------|
| `slate-bg` / `navy-950` | `#f8fafc` (Slate-50) | Main application background |
| `slate-card` / `navy-900` | `#ffffff` (White) | Cards, Sidebar, Panels |
| `navy-800` | `#e2e8f0` (Slate-200) | Borders, Dividers |

### Text Colors
| Class | Hex | Usage |
|-------|-----|-------|
| `text-slate-900` | `#0f172a` | Headings, Primary Text |
| `text-slate-800` | `#1e293b` | Body Text |
| `text-slate-500` | `#64748b` | Secondary Text, Meta info |

## Configuration Strategy

The configuration is injected via the `tailwind.config` script block in `app/templates/base.html`.

### Consistency Mapping
To ensure third-party components or legacy templates match the theme, we have mapped the standard `gray` palette to `slate`:

```javascript
colors: {
    gray: {
        50: "#f8fafc", // slate-50
        // ...
        900: "#0f172a" // slate-900
    }
}
```

This ensures that any usage of `bg-gray-50` or `text-gray-700` automatically adopts the system's `slate` tint.
