# Visual Style Guide

## 1. Layout & Structure
*   **Grid System**: 8px baseline grid. All spacing/padding should be multiples of 4px or 8px (e.g., `p-2` (8px), `p-4` (16px), `p-6` (24px), `p-8` (32px), `p-12` (48px)).
*   **Sidebar**: Fixed width (16rem/256px), White background (`bg-white`), right border (`border-slate-200`).
*   **Main Content**: Fluid width, Light Slate background (`bg-slate-50`).
*   **Cards**: White background, rounded corners (`rounded-xl` or `rounded-2xl`), light border (`border-slate-200`), subtle shadow (`shadow-sm`).
*   **Hover Effects**: Cards lift slightly (`-translate-y`) and increase shadow (`shadow-md`).
*   **Visual Hierarchy**: F-pattern layout for dashboards. Key metrics/actions at top left/center.

## 2. Typography
*   **Font Family**: `Noto Sans SC` (Chinese), `Manrope` (English/Numbers).
*   **Headings**:
    *   **H1**: 20-24px (`text-xl` to `text-2xl`), `font-bold` or `font-black`, `text-slate-900`.
    *   **H2**: 18-20px (`text-lg` to `text-xl`), `font-bold`, `text-slate-800`.
*   **Body**: 14-16px (`text-sm` to `text-base`), `text-slate-800`.
*   **Labels**: 12px (`text-xs`), `font-bold`, `uppercase`, `tracking-wider`, `text-slate-500`.

## 3. Interactive Elements

### Buttons
*   **Primary**: `bg-blue-600` (hover: `bg-blue-700`), text white, rounded.
*   **Secondary/Outline**: `bg-white`, `border border-gray-200`, text gray-600.
*   **Icon Buttons**: `text-slate-500` hover `text-slate-900`.
*   **Feedback**: All interactive elements must provide immediate visual feedback (hover, active, loading states).

### Form Inputs
*   **Background**: `bg-white` or `bg-gray-50`.
*   **Border**: `border-gray-200` (focus: `ring-2 ring-blue-500`).
*   **Text**: `text-sm`, `text-gray-800`.
*   **Smart Features**: Auto-completion, input formatting, memory.

## 4. Status Indicators
*   **Success**: `text-green-600`, `bg-green-50`.
*   **Warning**: `text-yellow-600`, `bg-yellow-50`.
*   **Error**: `text-red-600`, `bg-red-50`.
*   **Info**: `text-blue-600`, `bg-blue-50`.

## 5. Accessibility (WCAG)
*   **Contrast**: Text contrast ratio >= 4.5:1.
*   `text-slate-500` on `white` is acceptable for secondary text.
*   `text-slate-400` should only be used for disabled states or non-critical labels.
*   **ARIA**: Use semantic HTML tags.

## 6. Motion Design
*   **Duration**: 200ms - 500ms.
*   **Easing**: Standard cubic-bezier curves (e.g., `ease-in-out`, `ease-out`).
*   **Usage**: Hover states, modal appearances, toast notifications.
